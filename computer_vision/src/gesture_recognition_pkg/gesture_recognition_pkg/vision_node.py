#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Node de Visió per al Reconeixement de Gestos.

Aquest node s'encarrega d'analitzar les imatges rebudes des de la càmera
del robot, detectar la presència d'una mà utilitzant MediaPipe, extreure'n
les característiques (landmarks) i utilitzar un model de xarxa neuronal 
(EurekaNet) per predir el gest realitzat. Finalment, publica l'acció
corresponent perquè el robot l'executi.
"""

import os
import threading
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
import rclpy
import torch
import torch.nn as nn
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int8

# ==========================================
#         CONFIGURACIÓ GLOBAL
# ==========================================

# Classes reconegudes per defecte pel model
EUREKA_CLASSES = [
    'D0X', 'B0A', 'B0B', 'G01', 'G02', 'G03',
    'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11'
]

# Etiquetes de text per mostrar en pantalla
LABELS_EN = {
    "D0X": "Non-gesture",
    "B0A": "Pointing 1 finger",
    "B0B": "Pointing 2 fingers",
    "G01": "Click 1 finger",
    "G02": "Click 2 fingers",
    "G03": "Throw up",
    "G04": "Throw down",
    "G05": "Throw left",
    "G06": "Throw right",
    "G07": "Open twice",
    "G08": "Double click 1",
    "G09": "Double click 2",
    "G10": "Zoom in",
    "G11": "Zoom out",
}

# Paràmetres per al processament temporal i característiques
M_KEY_FRAMES = 15
NUM_LANDMARKS = 21
COMPONENTS = 2
INPUT_SIZE = (NUM_LANDMARKS ** 2) * COMPONENTS * (M_KEY_FRAMES - 1)
DEVICE = torch.device("cpu")

# ==========================================
#         LLINDARS (THRESHOLDS)
# ==========================================

CONF_THR_D0X = 0.75
MARGIN_D0X = 0.15

CONF_THR_GESTURE = 0.55
MARGIN_GESTURE = 0.07

FINGER_CLASSES = {"B0A", "B0B", "G01", "G02", "G08", "G09"}
CONF_THR_FINGER = 0.48
MARGIN_FINGER = 0.04

# Paràmetres del sistema de vots i bloqueig temporal
BUFFER_SIZE = 7
MIN_VOTES = 4

ENABLE_LOCK = True
LOCK_RELEASE_VOTES = 5
LOCK_BUFFER_SIZE = 7
MIN_VOTES_LOCK_FINGER = 3
MIN_VOTES_LOCK_DEFAULT = MIN_VOTES

# Mapa d'accions (traducció de Gest a ID d'acció per al TurtleBot)
GEST_MAPA = {
    'B0B': 0, 
    'B0A': 1, 
    'G02': 2, 
    'G04': 3, 
    'G05': 4,
    'G03': 5, 
    'G10': 6, 
    'G11': 7, 
    'G01': 8, 
    'G07': 9, 
    'G06': 10
}


class EurekaNet(nn.Module):
    """
    Model de Xarxa Neuronal per al reconeixement de gestos.
    Consta de capes lineals amb normalització i dropout.
    """
    def __init__(self):
        super(EurekaNet, self).__init__()
        
        self.fc1 = nn.Linear(INPUT_SIZE, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.25)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc3 = nn.Linear(64, len(EUREKA_CLASSES))

    def forward(self, x):
        """Passada cap endavant (forward pass) de la xarxa."""
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class NodeDeVisio(Node):
    """
    Classe principal del node ROS 2 encarregat de processar les imatges
    i predir els gestos utilitzant la xarxa neuronal i MediaPipe.
    """
    def __init__(self):
        super().__init__('node_de_visio')
        
        # Configuració de la qualitat de servei (QoS) per optimitzar la recepció d'imatges
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscriptor d'imatges comprimides
        self.sub = self.create_subscription(
            CompressedImage, 
            '/image_raw/compressed', 
            self.callback_imatge, 
            qos_profile
        )
        
        # Publicador de l'acció detectada
        self.pub = self.create_publisher(Int8, '/gesture_command', 10)
        
        # Pont per convertir imatges de ROS a format OpenCV
        self.pont = CvBridge()
        self.latest_processed_frame = None
        self.new_frame_available = False
        
        # Variables de depuració internes
        self._dbg_top1 = 0.0
        self._dbg_margin = 0.0

        # Carrega del model i pesos de la xarxa
        try:
            package_share_dir = get_package_share_directory('gesture_recognition_pkg')
            model_path = os.path.join(package_share_dir, 'models', 'eureka_model.pth')
            self.model = EurekaNet().to(DEVICE)
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.eval()
        except Exception as e:
            self.get_logger().error(f"Error carregant el model: {str(e)}")

        # Inicialització de MediaPipe per a la detecció de mans
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            model_complexity=0, 
            max_num_hands=1, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Buffers temporals (historials de dades i de respostes)
        self.history = deque(maxlen=M_KEY_FRAMES)
        self.votes = deque(maxlen=BUFFER_SIZE)
        
        # Control del bloqueig temporal per evitar oscil·lacions ràpides entre gestos
        self.locked_code = None
        self.lock_release_buffer = deque(maxlen=LOCK_BUFFER_SIZE)

    def calculate_features(self, buffer):
        """
        Calcula les característiques d'entrada al model a partir 
        de l'historial de punts (landmarks).
        """
        sel = np.array(buffer, dtype=np.float32)
        feats = []
        for t in range(1, sel.shape[0]):
            curr, prev = sel[t], sel[t - 1]
            diff = curr[:, np.newaxis, :] - prev[np.newaxis, :, :]
            feats.append(diff.reshape(-1))

        return np.concatenate(feats).astype(np.float32)

    def _get_thresholds(self, pred_code: str):
        """
        Obté els llindars de confiança i marge específics
        depenent de la classe de gest predit.
        """
        if pred_code == "D0X":
            return CONF_THR_D0X, MARGIN_D0X
        if pred_code in FINGER_CLASSES:
            return CONF_THR_FINGER, MARGIN_FINGER
        return CONF_THR_GESTURE, MARGIN_GESTURE

    def _predict_code(self, feats: np.ndarray) -> str:
        """
        Fa la predicció utilitzant les característiques i la xarxa neuronal.
        Filtra certs gestos segons llindars de confiança i marge.
        """
        if feats is None or feats.shape[0] != INPUT_SIZE:
            self._dbg_top1 = 0.0
            self._dbg_margin = 0.0
            return "Uncertain"

        inp = torch.tensor(feats).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self.model(inp)
            probs = torch.softmax(out, dim=1)
            
            # Obtenim els dos valors més probables per avaluar el marge i confiança
            top2 = torch.topk(probs, k=2, dim=1)
            conf1 = top2.values[0, 0].item()
            conf2 = top2.values[0, 1].item()
            pred_code = EUREKA_CLASSES[top2.indices[0, 0].item()]
            margin = conf1 - conf2
            
            self._dbg_top1 = conf1
            self._dbg_margin = margin

            # --- MODIFICACIÓ SOL·LICITADA ---
            # Eliminem G01 i G02 de les aturades i G08/G09 per no usats.
            # Els convertim tots en Non-gesture (D0X).
            if pred_code in ["G01", "G02", "G08", "G09"]:
                pred_code = "D0X"
            # -------------------------------

            conf_thr, mar_thr = self._get_thresholds(pred_code)
            
            # Només retorna el gest si supera els llindars exigits; sinó retorna "Uncertain"
            if conf1 > conf_thr and margin > mar_thr:
                return pred_code
            else:
                return "Uncertain"

    def _pick_winner(self, votes_deque: deque):
        """
        Selecciona el gest guanyador basat en l'historial de vots (majoria).
        Ignora "Uncertain", "NoGesture", "D0X" si hi ha altres opcions viables.
        """
        if not votes_deque:
            return "Uncertain", 0
            
        counts = Counter(votes_deque)
        real_counts = {
            k: v for k, v in counts.items() 
            if k not in ("Uncertain", "NoGesture", "D0X")
        }
        
        if real_counts:
            winner = max(real_counts.items(), key=lambda kv: kv[1])[0]
            return winner, real_counts[winner]
            
        # Si no hi ha cap gest "real" viable, retorna el més freqüent de totes maneres
        winner, count = counts.most_common(1)[0]
        return winner, count

    def callback_imatge(self, msg):
        """
        Funció callback que es crida cada vegada que arriba una imatge nova.
        S'encarrega d'orquestrar el procés, la detecció, predicció i publicació.
        """
        try:
            # Converteix el missatge ROS a imatge OpenCV compatible (RGB per Mediapipe)
            frame = self.pont.compressed_imgmsg_to_cv2(msg)
            res = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            pred_code = "Uncertain"

            if res.multi_hand_landmarks:
                # Si es detecten mans, s'extreuen i dibuixen els landmarks
                punts = res.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, punts, self.mp_hands.HAND_CONNECTIONS)
                self.history.append([[l.x, l.y] for l in punts.landmark])
                
                # Quan tenim suficients frames, procedim a fer la predicció
                if len(self.history) >= M_KEY_FRAMES:
                    feats = self.calculate_features(list(self.history))
                    pred_code = self._predict_code(feats)
            else:
                # Si no hi ha mà visible, buidem progressivament l'historial
                if self.history:
                    self.history.popleft()
                pred_code = "NoGesture"

            # Afegeix el resultat de la predicció a l'historial de vots
            self.votes.append(pred_code)
            winner, count = self._pick_winner(self.votes)
            
            # Gestió del bloqueig (Lock) per mantenir un gest si és estable
            if ENABLE_LOCK:
                self.lock_release_buffer.append(winner)
                if self.locked_code:
                    # Desbloqueja si s'han detectat "D0X" continus recentment
                    if Counter(self.lock_release_buffer).get("D0X", 0) >= LOCK_RELEASE_VOTES:
                        self.locked_code = None
                else:
                    # Bloqueja si hi ha suficients vots
                    min_v = MIN_VOTES_LOCK_FINGER if winner in FINGER_CLASSES else MIN_VOTES_LOCK_DEFAULT
                    if count >= min_v and winner not in ("Uncertain", "D0X", "NoGesture"):
                        self.locked_code = winner

            # Determina si cal dibuixar/publicar segons el bloqueig i el guanyador actual
            display_code = self.locked_code if (ENABLE_LOCK and self.locked_code) else winner
            should_publish = display_code not in ("Uncertain", "NoGesture", "D0X") and \
                             (self.locked_code or count >= MIN_VOTES)

            # Publicació de l'acció
            if should_publish:
                accio_id = GEST_MAPA.get(display_code, -1)
                if accio_id != -1:
                    msg_out = Int8()
                    msg_out.data = accio_id
                    self.pub.publish(msg_out)
                    
                    # Indicador visual de publicació exitosa
                    cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 255, 0), -1)

            # Dibuixar text overlay sobre la imatge
            overlay_text = LABELS_EN.get(display_code, display_code)
            color = (0, 255, 0) if should_publish else (200, 200, 200)
            cv2.rectangle(frame, (0, 0), (850, 95), (0, 0, 0), -1)
            cv2.putText(frame, overlay_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            # Actualitza els valors per a que el menú d'OPENCV pugui mostrar-ho
            self.latest_processed_frame = frame
            self.new_frame_available = True
            
        except Exception as e:
            self.get_logger().error(f"Error a callback_imatge: {e}")


def main(args=None):
    """
    Funció principal del node ROS.
    Inicialitza i executa el NodeDeVisio en un fil separat,
    i gestiona la finestra d'OpenCV en el fil principal per poder 
    veure les imatges del robot.
    """
    rclpy.init(args=args)
    node = NodeDeVisio()
    
    # S'inicia el node de ROS 2 en un fil separat
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    
    cv2.namedWindow("Node de visió", cv2.WINDOW_NORMAL)
    
    try:
        while rclpy.ok():
            # Actualitza la interfície si arriba una imatge nova
            if node.new_frame_available and node.latest_processed_frame is not None:
                cv2.imshow("Node de visió", node.latest_processed_frame)
                node.new_frame_available = False
                
            # 'q' per tancar la finestra
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Neteja de recursos
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


