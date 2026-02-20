#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Node de Control per al Robot TurtleBot.

Aquest node s'encarrega d'interpretar les ordres rebudes del node de visió
(Gestures) i traduir-les a comandes de velocitat per moure el robot. Inclou
diferents lògiques de navegació autònoma, entre elles:
- Seguiment de parets (autònom) utilitzant el LaserScan.
- Moviment bàsic (Endavant, Enrere, Girs de 90 graus).
- Seguretat per evitar xocs frontals automàticament.
"""

import math
import os
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int8


class ControlNode(Node):
    """
    Classe principal del node de control del TurtleBot que processa els gestos 
    i les dades del làser, i finalment publica la velocitat de les rodes.
    """
    def __init__(self):
        super().__init__('control_node')

        # ==========================================
        #         PARÀMETRES DEL NODE
        # ==========================================
        self.declare_parameter('gesture_topic', 'gesture_command')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        gesture_topic = self.get_parameter('gesture_topic').get_parameter_value().string_value
        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        # ==========================================
        #         SUBSCRIPTORS I PUBLICADORS
        # ==========================================
        self.sub_gest = self.create_subscription(Int8, gesture_topic, self.gest_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data)

        # Publicador de velocitats (TwistStamped utilitzat pel turtlebot local)
        self.pub_vel = self.create_publisher(TwistStamped, cmd_vel_topic, 10)

        # ==========================================
        #         MAPA DE GESTOS I IDS
        # ==========================================
        self.STOP_IDS = {0, 2, 8}      # G02, B0B, G01
        self.ID_REPEAT = 1            # B0A

        self.ID_BACKWARD = 3
        self.ID_TURN_LEFT = 4
        self.ID_FORWARD = 5
        self.ID_FOLLOW_RIGHT = 6
        self.ID_FOLLOW_LEFT = 7
        self.ID_ROTATE = 9
        self.ID_TURN_RIGHT = 10

        # ==========================================
        #         CONFIGURACIÓ DE MOVIMENT
        # ==========================================
        # Velocitats de desplaçament
        self.lin_speed = 0.2     # m/s
        self.ang_speed = 0.8     # rad/s

        # Temporitzadors estàtics d'accions predissenyades
        self.t_rotate = 8.0
        self.t_fwd = 5.0
        self.t_bwd = 5.0
        self.t_turn_90 = 2.0

        # Paràmetres per al seguiment de parets (Follow mode)
        self.wall_presence_dist = 0.50
        self.follow_forward_lin = 0.20
        self.follow_turn_ang = 0.50
        self.follow_recover_ang = 0.30
        self.follow_timeout_s = 30.0

        self.front_sector_deg = 20.0
        self.obst_front_dist = 0.35

        self.accept_after_t = 0.0
        self.post_action_cooldown = 0.6

        # ==========================================
        #         ESTAT INTERN DEL ROBOT
        # ==========================================
        self.last_executed_id = None
        self.rearmed = True
        self.last_action_id = None
        self.last_scan = None

        # Variables d'estímul síncron (accions temporitzades)
        self.action_active = False
        self.action_start_t = 0.0
        self.action_duration = 0.0
        self.action_lin = 0.0
        self.action_ang = 0.0

        # FSM per seguiment exclusiu
        self.mode_follow = None
        self.follow_state = None
        self.preferred_turn = "RIGHT"
        self.mode_follow_deadline_t = 0.0
        self.follow_log_period = 0.5
        self._last_follow_log_t = 0.0

        # Instrucció de repetir "Repeat Latch"
        self.repeat_latched = False
        self.last_repeat_msg_t = 0.0
        self.repeat_release_gap_s = 0.7

        # Temporitzador general que governa la màquina (a 20Hz aprox.)
        self.create_timer(0.05, self.loop)

        self.get_logger().info("CONTROL NODE LLEST I FUNCIONANT.")
        self.get_logger().info(f"Subscrit a gestos: '{gesture_topic}' i làser: '{scan_topic}'. Publicant a '{cmd_vel_topic}' (TwistStamped).")

    def scan_callback(self, msg: LaserScan):
        """Emmagatzema les dades més recents del làser."""
        self.last_scan = msg

    def _min_range_in_sector(self, scan: LaserScan, center_deg: float, half_width_deg: float) -> float:
        """
        Retorna la distància mínima detectada pel làser en un sector circular
        definit per un centre i una amplada (en graus).
        """
        if scan is None:
            return float('inf')

        center = math.radians(center_deg)
        half = math.radians(half_width_deg)

        amin = scan.angle_min
        ainc = scan.angle_increment
        n = len(scan.ranges)

        rmin = float('inf')
        for i in range(n):
            ang = amin + i * ainc
            if (center - half) <= ang <= (center + half):
                d = scan.ranges[i]
                if math.isfinite(d) and d > scan.range_min:
                    rmin = min(rmin, d)
        
        return rmin

    def _front_distance(self) -> float:
        """Calcula la distància lliure al front del robot."""
        return self._min_range_in_sector(
            self.last_scan, 
            center_deg=0.0, 
            half_width_deg=self.front_sector_deg
        )

    def _obstacle_ahead(self) -> bool:
        """Comprova si hi ha un obstacle just davant."""
        return self._front_distance() < self.obst_front_dist

    def _side_distance(self, side: str) -> float:
        """Busca distància al lateral especificat ('right' o 'left')."""
        if side == "right":
            return self._min_range_in_sector(self.last_scan, center_deg=-90.0, half_width_deg=15.0)
        return self._min_range_in_sector(self.last_scan, center_deg=90.0, half_width_deg=15.0)

    def _wall_present(self, side: str) -> bool:
        """Comprova si hi ha una paret al costat establert."""
        d = self._side_distance(side)
        return math.isfinite(d) and (d < self.wall_presence_dist)

    def _busy(self) -> bool:
        """Retorna True si el robot ja està efectuant una acció automàtica."""
        return self.action_active or (self.mode_follow is not None)

    def _arm_again(self):
        """Rearma l'accepte de nous gestos després d'estar ocupat o parat."""
        self.rearmed = True

    def _latch_execute(self, act_id: int):
        """Registra un gest com executat i prevé execucions duplicades sense rearmament."""
        self.last_executed_id = act_id
        self.rearmed = False

    def _stop_follow(self):
        """Atura les accions de seguiment i reinicia la màquina d'estats de paret."""
        self.mode_follow = None
        self.follow_state = None
        self.mode_follow_deadline_t = 0.0
        self.preferred_turn = "RIGHT"

    def _cancel_timed_action(self):
        """Atura qualsevol acció seqüencial temporal en curs."""
        self.action_active = False
        self.action_lin = 0.0
        self.action_ang = 0.0
        self.action_duration = 0.0

    def _cancel_all(self, reason: str):
        """Atura completament el moviment per qualsevol motiu."""
        self.get_logger().info(f"CANCEL·LAT: {reason}")
        self._stop_follow()
        self._cancel_timed_action()
        self.accept_after_t = time.monotonic() + self.post_action_cooldown
        self.send_vel(0.0, 0.0)

    def _start_timed_action(self, lin: float, ang: float, duration: float, name: str):
        """Inicia una acció síncrona de duració coneguda."""
        self._stop_follow()
        self.action_active = True
        self.action_start_t = time.monotonic()
        self.action_duration = float(duration)
        self.action_lin = float(lin)
        self.action_ang = float(ang)
        self.get_logger().info(f"START {name}: lin={lin:.2f} ang={ang:.2f} dur={duration:.1f}s")

    def _start_follow(self, side: str):
        """Inicia l'estat de seguir paret, aturant altres ordres."""
        self._cancel_timed_action()
        self.mode_follow = side
        self.follow_state = "IDLE"
        self.mode_follow_deadline_t = time.monotonic() + self.follow_timeout_s
        self.preferred_turn = "RIGHT"
        self.get_logger().info(f"FOLLOW {side.upper()} START (30s max)")

    def _repeat_last(self):
        """Repeteix la darrera acció vàlida enregistrada."""
        if self.last_action_id is None:
            self.get_logger().warn("REPEAT: no hi ha darrera acció (last_action).")
            return
        act = self.last_action_id
        self.get_logger().info(f"REPEAT last_action={act}")
        self.rearmed = True
        self._handle_action(act, from_repeat=True)

    def gest_callback(self, msg: Int8):
        """Processa el gest entrant interceptat pel tòpic corresponent."""
        act = int(msg.data)
        now = time.monotonic()

        # Aturada d'emergència en qualsevol moment
        if act in self.STOP_IDS:
            self._cancel_all(f"ATURADA D'EMERGÈNCIA (id={act})")
            self._arm_again()
            return

        # Ordre de repetir
        if act == self.ID_REPEAT:
            self.last_repeat_msg_t = now
            if (not self.repeat_latched) and (not self._busy()) and (now >= self.accept_after_t):
                self.repeat_latched = True
                self._repeat_last()
            return

        # Cooldown de silenci si s'ha acabat recentment
        if now < self.accept_after_t:
            return

        # Ignorar peticions si estem en marxa independentment
        if self._busy():
            return

        # Evitar auto-repetiment per culpa dels enviaments de 6 fps incontrolats
        if (self.last_executed_id is not None) and (act == self.last_executed_id) and (not self.rearmed):
            return

        self._handle_action(act, from_repeat=False)

    def _handle_action(self, act: int, from_repeat: bool = False):
        """Enllaça l'ID del gest detectat amb la seva funció corresponent."""
        if act == self.ID_FOLLOW_RIGHT:
            self._latch_execute(act)
            self._start_follow("right")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_FOLLOW_LEFT:
            self._latch_execute(act)
            self._start_follow("left")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_ROTATE:
            self._latch_execute(act)
            self._start_timed_action(0.0, +self.ang_speed, self.t_rotate, "ROTAR")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_FORWARD:
            self._latch_execute(act)
            self._start_timed_action(+self.lin_speed, 0.0, self.t_fwd, "AVANÇAR")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_BACKWARD:
            self._latch_execute(act)
            self._start_timed_action(-self.lin_speed, 0.0, self.t_bwd, "RETROCEDIR")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_TURN_LEFT:
            self._latch_execute(act)
            self._start_timed_action(0.0, -self.ang_speed, self.t_turn_90, "GIR ESQUERRA")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_TURN_RIGHT:
            self._latch_execute(act)
            self._start_timed_action(0.0, +self.ang_speed, self.t_turn_90, "GIR DRETA")
            if not from_repeat:
                self.last_action_id = act
            return

        self.get_logger().warn(f"Acció desconeguda {act} (ignorada)")

    def loop(self):
        """
        Bucle de control intern que comprova els temps, les accions temporitzades
        i la màquina d'estats (FSM) de manera periòdica.
        """
        now = time.monotonic()

        # Allibera el bloqueig de "Repeat" si el temps s'esgota
        if self.repeat_latched and (now - self.last_repeat_msg_t) > self.repeat_release_gap_s:
            self.repeat_latched = False

        # Mode de seguir parets (Comportament Autònom)
        if self.mode_follow in ("right", "left"):
            if now >= self.mode_follow_deadline_t:
                self.get_logger().info("FOLLOW TIMEOUT -> STOP")
                self._stop_follow()
                self.accept_after_t = time.monotonic() + self.post_action_cooldown
                self.send_vel(0.0, 0.0)
                return

            self._follow_fsm_step(self.mode_follow)
            self._log_follow(now, self.mode_follow)
            return

        # Execució d'Accions Temporitzades
        if self.action_active:
            elapsed = now - self.action_start_t
            if elapsed >= self.action_duration:
                self._cancel_timed_action()
                self.accept_after_t = time.monotonic() + self.post_action_cooldown
                self.send_vel(0.0, 0.0)
                return

            self.send_vel(self.action_lin, self.action_ang)
            return

        # Si no hi ha cap acció, garanteix estar completament parat
        self.send_vel(0.0, 0.0)

    def _follow_fsm_step(self, side: str):
        """
        Màquina d'Estats Finits (FSM) per el mode de seguir la paret.
        Estats disponibles: IDLE, DECIDE, TURNING, FOLLOW, RECOVER.
        """
        obstacle = self._obstacle_ahead()
        wall_left = self._wall_present("left")
        wall_right = self._wall_present("right")

        follow_side_present = wall_right if side == "right" else wall_left

        if self.follow_state == "IDLE":
            if obstacle:
                self.follow_state = "DECIDE"
                return
            self.send_vel(self.follow_forward_lin, 0.0)
            return

        if self.follow_state == "DECIDE":
            if (not wall_right) and wall_left:
                self.preferred_turn = "RIGHT"
            elif (not wall_left) and wall_right:
                self.preferred_turn = "LEFT"
            else:
                self.preferred_turn = "RIGHT"
            self.follow_state = "TURNING"
            return

        if self.follow_state == "TURNING":
            if not obstacle:
                self.follow_state = "FOLLOW"
                self.send_vel(self.follow_forward_lin, 0.0)
                return
            ang = -self.follow_turn_ang if self.preferred_turn == "RIGHT" else +self.follow_turn_ang
            self.send_vel(0.0, ang)
            return

        if self.follow_state == "FOLLOW":
            if obstacle:
                self.follow_state = "TURNING"
                return
            if not follow_side_present:
                self.follow_state = "RECOVER"
                return
            self.send_vel(self.follow_forward_lin, 0.0)
            return

        if self.follow_state == "RECOVER":
            if obstacle:
                self.follow_state = "TURNING"
                return
            if follow_side_present:
                self.follow_state = "FOLLOW"
                self.send_vel(self.follow_forward_lin, 0.0)
                return

            ang = -self.follow_recover_ang if side == "right" else +self.follow_recover_ang
            self.send_vel(self.follow_forward_lin, ang)

    def _log_follow(self, now: float, side: str):
        """LogPeriòdic de la telemetria per a depurar la FSM i els sensors."""
        if now - self._last_follow_log_t < self.follow_log_period:
            return
            
        self._last_follow_log_t = now

        front = self._front_distance()
        dl = self._side_distance("left")
        dr = self._side_distance("right")
        wl = self._wall_present("left")
        wr = self._wall_present("right")
        remaining = max(0.0, self.mode_follow_deadline_t - now)

        self.get_logger().info(
            f"FOLLOW[{side}] st={self.follow_state} front={front:.2f} "
            f"L={dl:.2f}({int(wl)}) R={dr:.2f}({int(wr)}) t_left={remaining:.1f}"
        )

    def send_vel(self, lin: float, ang: float):
        """
        Llença comandes de velocitat a les rodes.
        S'usa TwistStamped amb una capçalera neutra perquè és el que
        el Turtlebot acceptava prèviament de forma nativa.
        """
        m = TwistStamped()
        m.header.stamp.sec = 0
        m.header.stamp.nanosec = 0
        m.header.frame_id = ''
        m.twist.linear.x = float(lin)
        m.twist.angular.z = float(ang)
        self.pub_vel.publish(m)


def main(args=None):
    """
    Funció principal de llançament i terminació segura.
    Assegura que sempre enviem velocitat zero en apagar el node.
    """
    rclpy.init(args=args)
    node = ControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Garanteix que el robot s'atura quan es tanca el node.
        stop = TwistStamped()
        stop.header.stamp.sec = 0
        stop.header.stamp.nanosec = 0
        stop.twist.linear.x = 0.0
        stop.twist.angular.z = 0.0
        node.pub_vel.publish(stop)
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


