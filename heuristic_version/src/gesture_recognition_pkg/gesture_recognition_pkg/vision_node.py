#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Node de Visió (Versió Heurística) per al Reconeixement de Gestos.

Aquest node s'encarrega d'analitzar les imatges rebudes des de la càmera
del robot, detectar la presència d'una mà utilitzant MediaPipe, extreure'n
les característiques (landmarks) i avaluar condicions geomètriques 
(heurístiques) per identificar tant gestos estàtics com dinàmics.
"""

import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8

# ==========================================
#         DEFINICIÓ DE GESTOS
# ==========================================

# Codis de gestos estàtics (basats en posició fixa dits)
GEST_PARAR      = 0  # puny
GEST_ENDAVANT   = 1  # mà oberta -> exploració endavant segura
GEST_PARET      = 2  # polze amunt -> seguir paret
GEST_PATRULLA   = 3  # dos dits -> mini patrulla local

# Codis de gestos dinàmics (basats en moviment temporal)
GEST_REPETIR    = 10  # gest circular amb el dit -> repetir última acció

# Diccionari visual per a depuració en pantalla
NOMS_GESTOS = {
    GEST_PARAR:    "puny (aturar)",
    GEST_ENDAVANT: "ma oberta (explorar endavant)",
    GEST_PARET:    "polze amunt (seguir paret)",
    GEST_PATRULLA: "dos dits (patrulla)",
    GEST_REPETIR:  "cercle (repetir accio)",
    -1:            "..."
}


class NodeDeVisio(Node):
    """
    Node principal encarregat de subscriure's a la càmera, processar cada 
    frame amb MediaPipe i publicar les accions corresponents a ROS 2.
    """

    def __init__(self):
        super().__init__('node_de_visio')

        # ==========================================
        #         PARÀMETRES DEL NODE
        # ==========================================
        self.declare_parameter('topic_imatge', '/camera/image_raw')
        self.declare_parameter('topic_ordres', '/gesture_command')
        
        topic_imatge = self.get_parameter('topic_imatge').get_parameter_value().string_value
        topic_ordres = self.get_parameter('topic_ordres').get_parameter_value().string_value

        # Publicador de comandes generades
        self.publicador_ordres = self.create_publisher(Int8, topic_ordres, 10)

        # Configuració del receptor de vídeo (QoS de millor esforç pel flux d'imatges)
        qos_video = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.pont = CvBridge()
        self.subscriptor_imatge = self.create_subscription(
            Image, 
            topic_imatge, 
            self.callback_imatge, 
            qos_video
        )

        # ==========================================
        #         INICIALITZACIÓ MEDIAPIPE
        # ==========================================
        self.mp_mans = mp.solutions.hands
        self.detector_mans = self.mp_mans.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_dibuix = mp.solutions.drawing_utils

        # ==========================================
        #         TEMPORITZADORS I BUFFERS (LÒGICA)
        # ==========================================
        # Filtre anti-spam / Rate-limiter (màxim 10 Hz)
        self.ultim_gest_publicat = -1
        self.temps_ultima_publicacio = 0.0
        self.periode_minim_publicacio = 0.10

        # Buffer circular d'historial per al gest dinàmic (Cercle referenciat pel dit índex)
        self.index_trail = deque(maxlen=25)
        self.temps_ultim_cercle = 0.0
        self.temps_minim_entre_cercles = 1.0     # 1 segon de cooldown per cercle

        self.get_logger().info(
            f'Node de visió llest. Mirant {topic_imatge} i publicant a {topic_ordres}.'
        )

    def callback_imatge(self, missatge: Image):
        """
        Es crida amb cada imatge rebruda de la càmera.
        Tradueix la imatge i processa la geometria de la mà.
        """
        try:
            frame = self.pont.imgmsg_to_cv2(missatge, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'Error al convertir la imatge: {e}')
            return

        # Adaptem i passem al motor de MediaPipe
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultats = self.detector_mans.process(rgb)

        gest_estatic = -1
        gest_final = -1

        if resultats.multi_hand_landmarks:
            punts_ma = resultats.multi_hand_landmarks[0]
            
            # Dibuixem els landmarks localitzats sobre el frame per la visualització d'Opencv
            self.mp_dibuix.draw_landmarks(frame, punts_ma, self.mp_mans.HAND_CONNECTIONS)

            # Classificació estructural per saber quins dits estan rectes o doblegats
            gest_estatic = self.classifica_gest_estatic(punts_ma)

            # Enregistrament de la posició del dit per habilitar gestos dinàmics
            self.actualitza_trail_index(punts_ma)
            gest_dinamic = self.detecta_cercle()

            # Lògica de prioritats (Gest dinàmic supera l'estàtic)
            if gest_dinamic == GEST_REPETIR:
                gest_final = GEST_REPETIR
            else:
                gest_final = gest_estatic
        else:
            # Si desapareix de càmera, reinicia la traçabilitat del dit
            self.index_trail.clear()

        # ==========================================
        #         LÒGICA DE PUBLICACIÓ (ANTI-SPAM)
        # ==========================================
        ara = time.time()
        if gest_final != -1 and (
            gest_final != self.ultim_gest_publicat
            or (ara - self.temps_ultima_publicacio) >= self.periode_minim_publicacio
        ):
            msg = Int8()
            msg.data = gest_final
            self.publicador_ordres.publish(msg)
            
            self.ultim_gest_publicat = gest_final
            self.temps_ultima_publicacio = ara

        # ==========================================
        #         FEEDBACK VISUAL A PANTALLA
        # ==========================================
        text_gest = NOMS_GESTOS.get(gest_final if gest_final != -1 else gest_estatic, "...")
        cv2.putText(frame, text_gest, (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        try:
            cv2.imshow('Control per Gestos (Heuristic)', frame)
            cv2.waitKey(1)
        except cv2.error:
            pass

    def classifica_gest_estatic(self, punts_ma) -> int:
        """
        Extreu les distàncies entre punts estratègics de la mà (landmarks)
        per determinar si estan estesos o doblegats i retorna l'ID del gest.
        """
        punts = punts_ma.landmark

        def dist(p1, p2):
            return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

        punta_polze = punts[self.mp_mans.HandLandmark.THUMB_TIP]
        punta_index = punts[self.mp_mans.HandLandmark.INDEX_FINGER_TIP]
        punta_cor = punts[self.mp_mans.HandLandmark.MIDDLE_FINGER_TIP]
        punta_anular = punts[self.mp_mans.HandLandmark.RING_FINGER_TIP]
        punta_petit = punts[self.mp_mans.HandLandmark.PINKY_TIP]

        canell = punts[self.mp_mans.HandLandmark.WRIST]
        base_index = punts[self.mp_mans.HandLandmark.INDEX_FINGER_MCP]

        artell_index = punts[self.mp_mans.HandLandmark.INDEX_FINGER_PIP]
        artell_cor = punts[self.mp_mans.HandLandmark.MIDDLE_FINGER_PIP]
        artell_anular = punts[self.mp_mans.HandLandmark.RING_FINGER_PIP]
        artell_petit = punts[self.mp_mans.HandLandmark.PINKY_PIP]

        # ---------------------------------------------------------------------
        # 1) Polze amunt -> GEST_PARET (Seguiment de paret)
        # ---------------------------------------------------------------------
        puntes_dits_avall = (
            punta_index.y > artell_index.y and
            punta_cor.y > artell_cor.y and
            punta_anular.y > artell_anular.y and
            punta_petit.y > artell_petit.y
        )
        if punta_polze.y < base_index.y and puntes_dits_avall:
            return GEST_PARET

        # ---------------------------------------------------------------------
        # 2) Mà oberta -> GEST_ENDAVANT (dits estirats lluny del canell)
        # ---------------------------------------------------------------------
        llindar_ma_oberta = dist(canell, base_index) * 1.8
        if (dist(canell, punta_polze) > llindar_ma_oberta * 0.8 and
            dist(canell, punta_index) > llindar_ma_oberta and
            dist(canell, punta_cor) > llindar_ma_oberta and
            dist(canell, punta_anular) > llindar_ma_oberta and
            dist(canell, punta_petit) > llindar_ma_oberta):
            return GEST_ENDAVANT

        # ---------------------------------------------------------------------
        # 3) Puny tancat -> GEST_PARAR
        # ---------------------------------------------------------------------
        llindar_puny = dist(canell, base_index) * 1.2
        if (dist(canell, punta_index) < llindar_puny and
            dist(canell, punta_cor) < llindar_puny and
            dist(canell, punta_anular) < llindar_puny and
            dist(canell, punta_petit) < llindar_puny):
            return GEST_PARAR

        # ---------------------------------------------------------------------
        # 4) Dos dits (índex + cor amunt) -> GEST_PATRULLA
        # ---------------------------------------------------------------------
        if (punta_index.y < artell_index.y and
            punta_cor.y   < artell_cor.y   and
            punta_anular.y > artell_anular.y and
            punta_petit.y  > artell_petit.y):
            return GEST_PATRULLA

        return -1

    def actualitza_trail_index(self, punts_ma):
        """Afegeix l'última coordenada X, Y del dit índex per analitzar trajectòries."""
        punts = punts_ma.landmark
        idx = self.mp_mans.HandLandmark.INDEX_FINGER_TIP
        p = punts[idx]
        self.index_trail.append((p.x, p.y))

    def detecta_cercle(self) -> int:
        """
        Calcula estadísticament si el recorregut guardat del dit índex 
        ocupa l'estructura matemàtica aproximada d'un cercle complet al pla 2D.
        """
        ara = time.time()
        
        # Requereix un mínim de mostres en el buffer
        if len(self.index_trail) < 10:
            return -1

        # Preveu multi-deteccions simultànies del mateix cercle (cooldown)
        if ara - self.temps_ultim_cercle < self.temps_minim_entre_cercles:
            return -1

        pts = np.array(self.index_trail, dtype=np.float32)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        rel = pts - np.array([cx, cy])

        radis = np.linalg.norm(rel, axis=1)
        radi_mig = np.mean(radis)
        
        # El cercle detectat no ha de ser massa petit sinó pot ser tremolor (noise)
        if radi_mig < 0.02:
            return -1
            
        # Punts massa heterogenis en distància del centre per conformar un cercle sòlid
        if np.std(radis) > radi_mig * 0.4:
            return -1

        angles = np.arctan2(rel[:, 1], rel[:, 0])
        diffs = np.diff(angles)
        
        # Ajust i acumulació dels girs suprimint el salt polar
        diffs = (diffs + math.pi) % (2 * math.pi) - math.pi
        gir_total = np.sum(np.abs(diffs))

        # Detecta com a vàlid si els angles sumats excedeixen 360 graus (2 Pi)
        if gir_total > 2 * math.pi:
            self.temps_ultim_cercle = ara
            return GEST_REPETIR

        return -1

    def destroy_node(self):
        """Tancament unificat que garanteix tancar també l'Opencv internament."""
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    """
    Funció principal del Node. 
    Inicialitza, mantén viva l'escolta (spin), i tanca adequadament.
    """
    rclpy.init(args=args)
    node = NodeDeVisio()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Tancant el node de visió heuristic...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
