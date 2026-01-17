#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import mediapipe as mp
import numpy as np

# --- codis de gest estàtics ---
GEST_PARAR      = 0  # puny
GEST_ENDAVANT   = 1  # mà oberta -> exploració endavant segura
GEST_PARET      = 2  # polze amunt -> seguir paret
GEST_PATRULLA   = 3  # dos dits -> mini patrulla local

# --- codi de gest dinàmic ---
GEST_REPETIR    = 10  # gest circular amb el dit -> repetir última acció

# --- noms bonics per pantalla / debugging ---
NOMS_GESTOS = {
    GEST_PARAR:    "puny (aturar)",
    GEST_ENDAVANT: "mà oberta (explorar endavant)",
    GEST_PARET:    "polze amunt (seguir paret)",
    GEST_PATRULLA: "dos dits (patrulla)",
    GEST_REPETIR:  "cercle (repetir acció)",
    -1:            "..."
}


class NodeDeVisio(Node):
    """
    Node de visió:
    1. Llegeix imatges de /camera/image_raw.
    2. Detecta mans amb MediaPipe.
    3. Classifica gestos estàtics (0–3) i un gest dinàmic (cercle -> 10).
    4. Publica Int8 al tòpic /gesture_command.
    """

    def __init__(self):
        super().__init__('node_de_visio')

        # paràmetres de noms de tòpics
        self.declare_parameter('topic_imatge', '/camera/image_raw')
        self.declare_parameter('topic_ordres', '/gesture_command')
        topic_imatge = self.get_parameter('topic_imatge').get_parameter_value().string_value
        topic_ordres = self.get_parameter('topic_ordres').get_parameter_value().string_value

        # publicador de gestos
        self.publicador_ordres = self.create_publisher(Int8, topic_ordres, 10)

        # subscripció a imatge amb QoS de vídeo
        qos_video = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.pont = CvBridge()
        self.subscriptor_imatge = self.create_subscription(
            Image, topic_imatge, self.callback_imatge, qos_video
        )

        # configuració mediapipe hands
        self.mp_mans = mp.solutions.hands
        self.detector_mans = self.mp_mans.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_dibuix = mp.solutions.drawing_utils

        # anti-spam per publicació
        self.ultim_gest_publicat = -1
        self.temps_ultima_publicacio = 0.0
        self.periode_minim_publicacio = 0.10  # màxim 10 Hz

        # buffer per al gest dinàmic (cercle amb el dit)
        self.index_trail = deque(maxlen=25)      # últimes posicions del dit índex
        self.temps_ultim_cercle = 0.0
        self.temps_minim_entre_cercles = 1.0     # 1 segon de "cooldown"

        self.get_logger().info(
            f'Node de visió llest. Mirant {topic_imatge} i publicant a {topic_ordres}.'
        )

    # ---------------------------------------------------------------------
    # callback d'imatge
    # ---------------------------------------------------------------------
    def callback_imatge(self, missatge: Image):
        try:
            frame = self.pont.imgmsg_to_cv2(missatge, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'error al convertir la imatge: {e}')
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultats = self.detector_mans.process(rgb)

        gest_estatic = -1
        gest_final = -1

        if resultats.multi_hand_landmarks:
            punts_ma = resultats.multi_hand_landmarks[0]
            self.mp_dibuix.draw_landmarks(frame, punts_ma, self.mp_mans.HAND_CONNECTIONS)

            # classifiquem gest estàtic
            gest_estatic = self.classifica_gest_estatic(punts_ma)

            # actualitzem el trail del dit índex per al gest dinàmic
            self.actualitza_trail_index(punts_ma)

            # intentem detectar gest dinàmic de cercle
            gest_dinamic = self.detecta_cercle()

            # prioritat: si hi ha cercle valid -> GEST_REPETIR
            if gest_dinamic == GEST_REPETIR:
                gest_final = GEST_REPETIR
            else:
                gest_final = gest_estatic
        else:
            # si no hi ha mà detectada, buidem el trail
            self.index_trail.clear()

        # --- publicació amb anti-spam ---
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

        # text per pantalla
        text_gest = NOMS_GESTOS.get(gest_final if gest_final != -1 else gest_estatic, "...")
        cv2.putText(frame, text_gest, (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        try:
            cv2.imshow('Control per Gestos', frame)
            cv2.waitKey(1)
        except cv2.error:
            pass

    # ---------------------------------------------------------------------
    # GESTOS ESTÀTICS
    # ---------------------------------------------------------------------
    def classifica_gest_estatic(self, punts_ma) -> int:
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

        # 1) polze amunt -> GEST_PARET (seguiment de paret)
        puntes_dits_avall = (
            punta_index.y > artell_index.y and
            punta_cor.y > artell_cor.y and
            punta_anular.y > artell_anular.y and
            punta_petit.y > artell_petit.y
        )
        if punta_polze.y < base_index.y and puntes_dits_avall:
            return GEST_PARET

        # 2) mà oberta -> GEST_ENDAVANT (tots els dits estirats lluny del canell)
        llindar_ma_oberta = dist(canell, base_index) * 1.8
        if (dist(canell, punta_polze) > llindar_ma_oberta * 0.8 and
            dist(canell, punta_index) > llindar_ma_oberta and
            dist(canell, punta_cor) > llindar_ma_oberta and
            dist(canell, punta_anular) > llindar_ma_oberta and
            dist(canell, punta_petit) > llindar_ma_oberta):
            return GEST_ENDAVANT

        # 3) puny tancat -> GEST_PARAR
        llindar_puny = dist(canell, base_index) * 1.2
        if (dist(canell, punta_index) < llindar_puny and
            dist(canell, punta_cor) < llindar_puny and
            dist(canell, punta_anular) < llindar_puny and
            dist(canell, punta_petit) < llindar_puny):
            return GEST_PARAR

        # 4) dos dits (índex + cor amunt) -> GEST_PATRULLA
        if (punta_index.y < artell_index.y and
            punta_cor.y   < artell_cor.y   and
            punta_anular.y > artell_anular.y and
            punta_petit.y  > artell_petit.y):
            return GEST_PATRULLA

        # ja no hi ha gest d'"un dit": no el classifiquem

        return -1

    # ---------------------------------------------------------------------
    # GEST DINÀMIC: CERCLE AMB EL DIT (REPETIR)
    # ---------------------------------------------------------------------
    def actualitza_trail_index(self, punts_ma):
        punts = punts_ma.landmark
        idx = self.mp_mans.HandLandmark.INDEX_FINGER_TIP
        p = punts[idx]
        self.index_trail.append((p.x, p.y))

    def detecta_cercle(self) -> int:
        """
        Detecta si el moviment recent del dit índex descriu aproximadament un cercle.
        Si sí, retorna GEST_REPETIR, sinó -1.
        """
        ara = time.time()
        if len(self.index_trail) < 10:
            return -1

        if ara - self.temps_ultim_cercle < self.temps_minim_entre_cercles:
            return -1

        pts = np.array(self.index_trail, dtype=np.float32)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        rel = pts - np.array([cx, cy])

        radis = np.linalg.norm(rel, axis=1)
        radi_mig = np.mean(radis)
        if radi_mig < 0.02:
            return -1
        if np.std(radis) > radi_mig * 0.4:
            return -1

        angles = np.arctan2(rel[:, 1], rel[:, 0])
        diffs = np.diff(angles)
        diffs = (diffs + math.pi) % (2 * math.pi) - math.pi
        gir_total = np.sum(np.abs(diffs))

        if gir_total > 2 * math.pi:
            self.temps_ultim_cercle = ara
            return GEST_REPETIR

        return -1

    # ---------------------------------------------------------------------
    # tancament
    # ---------------------------------------------------------------------
    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NodeDeVisio()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Tancant el node de visió...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
