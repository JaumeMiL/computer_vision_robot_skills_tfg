# Importem les llibreries necessàries
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8  # Missatge per enviar la comanda (0, 1, 2, 3, 4)
import cv2                     # OpenCV per a la gestió de la càmera
import mediapipe as mp         # MediaPipe per a la detecció de mans
import numpy as np

# Definim els codis dels gestos per a més claredat
GESTURE_STOP = 0
GESTURE_FORWARD = 1
GESTURE_BACKWARD = 2
GESTURE_TURN_LEFT = 3
GESTURE_TURN_RIGHT = 4

class VisionNode(Node):
    """
    Aquest node captura vídeo de la càmera, detecta gestos de la mà amb MediaPipe
    i publica una comanda simple en el tòpic /gesture_command.
    """
    def __init__(self):
        super().__init__('vision_node')
        
        # 1. Creador del publicador (Publisher)
        # Publicarà missatges de tipus Int8 al tòpic /gesture_command
        self.publisher_ = self.create_publisher(Int8, '/gesture_command', 10)
        
        # 2. Temporitzador (Timer)
        # Executarà la funció 'timer_callback' 30 cops per segon (approx. 30 FPS)
        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # 3. Configuració de la Càmera i MediaPipe
        self.cap = cv2.VideoCapture("/dev/video10", cv2.CAP_V4L2) # Inicia la captura de vídeo (càmera 0)
        if not self.cap.isOpened():
            self.get_logger().error('No s ha pogut obrir la càmera.')
            rclpy.shutdown()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,                # Només detectem una mà
            min_detection_confidence=0.7,   # Confiança mínima per a la detecció
            min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.get_logger().info('Node de Visió iniciat i llest.')

    def timer_callback(self):
        """
        Aquesta funció s'executa a cada tick del temporitzador.
        Llegeix un frame de la càmera, el processa i publica el gest detectat.
        """
        success, frame = self.cap.read()
        if not success:
            return

        # Girem el frame horitzontalment (efecte mirall) per a una interacció més natural
        frame = cv2.flip(frame, 1)
        
        # Convertim el color de BGR (OpenCV) a RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processem el frame amb MediaPipe
        results = self.hands.process(rgb_frame)
        
        command_msg = Int8()
        gesture = -1 # Valor per defecte si no es detecta cap gest

        # Si es detecten mans...
        if results.multi_hand_landmarks:
            # Dibuixem els punts i connexions de la mà sobre el frame
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Classifiquem el gest
                gesture = self.classify_gesture(hand_landmarks)

        if gesture != -1:
            command_msg.data = gesture
            self.publisher_.publish(command_msg)
            # Mostrem el gest reconegut a la pantalla
            cv2.putText(frame, f'GEST: {gesture}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Mostrem el vídeo en una finestra
        cv2.imshow("Gesture Control Vision", frame)
        cv2.waitKey(1)

    def classify_gesture(self, hand_landmarks):
        """
        Analitza les posicions dels punts de la mà (landmarks) per determinar el gest.
        Retorna el codi numèric del gest.
        """
        landmarks = hand_landmarks.landmark
        
        # Funció auxiliar per calcular la distància entre dos punts
        def get_distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

        # Obtenim la posició de les puntes dels dits i les articulacions principals
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # LÒGICA DE CLASSIFICACIÓ DE GESTOS 🖐️
        
        # 1. GEST: Polze amunt (Enrere)
        # - Punta del polze està per sobre de la base dels altres dits.
        # - Les puntes dels altres 4 dits estan per sota de les seves articulacions mitjanes.
        finger_tips_down = (index_tip.y > landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                            middle_tip.y > landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                            ring_tip.y > landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y and
                            pinky_tip.y > landmarks[self.mp_hands.HandLandmark.PINKY_FINGER_PIP].y)

        if thumb_tip.y < index_mcp.y and finger_tips_down:
            return GESTURE_BACKWARD

        # 2. GEST: Palma oberta (Endavant)
        # - Els 5 dits estan estesos. Comprovem que la punta de cada dit estigui lluny del canell.
        palm_open_threshold = get_distance(wrist, index_mcp) * 1.8
        if (get_distance(wrist, thumb_tip) > palm_open_threshold * 0.8 and
            get_distance(wrist, index_tip) > palm_open_threshold and
            get_distance(wrist, middle_tip) > palm_open_threshold and
            get_distance(wrist, ring_tip) > palm_open_threshold and
            get_distance(wrist, pinky_tip) > palm_open_threshold):
            return GESTURE_FORWARD

        # 3. GEST: Puny tancat (Aturar)
        # - La punta de tots els dits està a prop del canell (o de la base de la mà).
        fist_threshold = get_distance(wrist, index_mcp) * 1.2
        if (get_distance(wrist, index_tip) < fist_threshold and
            get_distance(wrist, middle_tip) < fist_threshold and
            get_distance(wrist, ring_tip) < fist_threshold and
            get_distance(wrist, pinky_tip) < fist_threshold):
            return GESTURE_STOP

        # 4. GEST: Girar Esquerra (dos dits amunt)
        # - Índex i cor amunt, els altres avall.
        if (index_tip.y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_tip.y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y > landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y > landmarks[self.mp_hands.HandLandmark.PINKY_FINGER_PIP].y):
            return GESTURE_TURN_LEFT
            
        # 5. GEST: Girar Dreta (un dit amunt - índex)
        # - Només l'índex amunt.
        if (index_tip.y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_tip.y > landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y > landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y > landmarks[self.mp_hands.HandLandmark.PINKY_FINGER_PIP].y):
            return GESTURE_TURN_RIGHT
            
        return -1 # Cap gest reconegut


def main(args=None):
    rclpy.init(args=args)
    vision_node = VisionNode()
    try:
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Destruïm el node i alliberem recursos
        vision_node.cap.release()
        cv2.destroyAllWindows()
        vision_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
