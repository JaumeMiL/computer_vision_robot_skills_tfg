import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
from sensor_msgs.msg import LaserScan
import numpy as np

# codis de gest (coherents amb vision_node.py)
GEST_PARAR      = 0   # puny
GEST_ENDAVANT   = 1   # mà oberta -> exploració endavant segura
GEST_PARET      = 2   # polze amunt -> seguir paret
GEST_PATRULLA   = 3   # dos dits -> mini patrulla
GEST_REPETIR    = 10  # cercle -> repetir última acció

# arm_command: convencions senzilles
ARM_NEUTRE = 0
ARM_OK     = 1
ARM_FAIL   = -1


class ControlNode(Node):
    """
    Node de control d'alt nivell:
    - rep gestos (Int8) de /gesture_command
    - segons el gest, inicia una acció complexa (explorar, patrullar, seguir paret...)
    - cada acció mira el LaserScan per evitar col·lisions
    - en acabar, envia una ordre al braç (/arm_command) per indicar èxit o fallada
    """

    def __init__(self):
        super().__init__('control_node')

        # subscripció a gestos
        self.subscription = self.create_subscription(
            Int8,
            '/gesture_command',
            self.gesture_callback,
            10
        )

        # subscripció a làser
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_pub = self.create_publisher(Int8, '/arm_command', 10)

        # estat del làser
        self.last_scan = None
        self.last_scan_time = self.get_clock().now()

        # FSM d'acció
        self.current_action = None          # codi de gest que s'està executant
        self.action_start_time = None
        self.action_state = None            # sub-estat intern
        self.last_successful_action = None  # per a GEST_REPETIR

        # paràmetres bàsics
        self.safe_front_dist = 0.4
        self.max_action_duration = 8.0      # segons

        # patrulla
        self.patrol_segment_index = 0

        # wall-follow
        self.wall_follow_start_time = None
        self.wall_follow_duration = 6.0     # segons
        self.target_wall_dist = 0.6
        self.max_wall_dist = 1.5

        # timer de control (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Node de Control iniciat i llest.')

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()

    def gesture_callback(self, msg: Int8):
        gest = msg.data
        self.get_logger().info(f'Rebut gest: {gest}')

        # GEST_PARAR: cancel·lar tota acció immediatament
        if gest == GEST_PARAR:
            self.stop_robot()
            self.finish_action(success=True)  # aturada "correcta"
            return

        # gest dinàmic: repetir última acció
        if gest == GEST_REPETIR:
            if self.last_successful_action is not None:
                self.get_logger().info(f'Repetint acció {self.last_successful_action}')
                self.start_action(self.last_successful_action)
            else:
                self.get_logger().warn('No hi ha cap acció prèvia per repetir.')
            return

        # per altres gestos:
        # si hi ha una acció en curs, la substituïm per la nova
        self.start_action(gest)

    # ------------------------------------------------------------------
    # gestió d'accions
    # ------------------------------------------------------------------
    def start_action(self, gest: int):
        self.get_logger().info(f'Iniciant acció per al gest {gest}')
        self.current_action = gest
        self.action_start_time = time.time()
        self.action_state = 0

        if gest == GEST_PATRULLA:
            self.patrol_segment_index = 0
        elif gest == GEST_PARET:
            self.wall_follow_start_time = time.time()

        # braç en posició neutra al començar
        self.send_arm_command(ARM_NEUTRE)

    def finish_action(self, success: bool):
        """
        Marca una acció com acabada, atura el robot i mou el braç.
        """
        if self.current_action is None:
            return

        self.stop_robot()

        if success:
            self.get_logger().info(f"Acció {self.current_action} completada amb èxit.")
            self.last_successful_action = self.current_action
            self.send_arm_command(ARM_OK)   # estendre braç "acció feta"
        else:
            self.get_logger().warn(f"Acció {self.current_action} NO s'ha pogut completar.")
            self.send_arm_command(ARM_FAIL) # gest de "no puc"

        self.current_action = None
        self.action_start_time = None
        self.action_state = None

    # ------------------------------------------------------------------
    # utilitats
    # ------------------------------------------------------------------
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def send_arm_command(self, state: int):
        msg = Int8()
        msg.data = state
        self.arm_pub.publish(msg)

    def get_front_min_distance(self):
        """
        Retorna la distància mínima aproximada al davant.
        """
        if self.last_scan is None:
            return None

        ranges = np.array(self.last_scan.ranges, dtype=float)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        angle_min = self.last_scan.angle_min
        angle_inc = self.last_scan.angle_increment
        angles = angle_min + np.arange(len(ranges)) * angle_inc

        # sector frontal +- 20 graus
        mask_front = np.logical_and(angles > -math.radians(20),
                                    angles <  math.radians(20))

        front_ranges = ranges[mask_front]
        if front_ranges.size == 0:
            return None
        return float(np.min(front_ranges))

    def get_side_distance(self, side: str = "right"):
        """
        Retorna la distància mínima aproximada a la paret lateral
        (side = 'right' o 'left').
        """
        if self.last_scan is None:
            return None

        ranges = np.array(self.last_scan.ranges, dtype=float)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        angle_min = self.last_scan.angle_min
        angle_inc = self.last_scan.angle_increment
        angles = angle_min + np.arange(len(ranges)) * angle_inc

        if side == "right":
            # sector lateral dret (-120º a -60º)
            mask = np.logical_and(angles < -math.radians(60),
                                  angles > -math.radians(120))
        else:
            # lateral esquerre (60º a 120º)
            mask = np.logical_and(angles > math.radians(60),
                                  angles < math.radians(120))

        side_ranges = ranges[mask]
        if side_ranges.size == 0:
            return None
        return float(np.min(side_ranges))

    # ------------------------------------------------------------------
    # bucle principal de control
    # ------------------------------------------------------------------
    def control_loop(self):
        # si no hi ha acció activa, no cal fer res
        if self.current_action is None:
            return

        # si l'acció dura massa: la cancel·lem per seguretat
        if self.action_start_time is not None:
            if time.time() - self.action_start_time > self.max_action_duration:
                self.get_logger().warn("Temps màxim d'acció excedit, aturant.")
                self.finish_action(success=False)
                return

        # si no tenim làser, no ens movem
        if self.last_scan is None:
            self.get_logger().warn_throttle(2.0, "Sense LaserScan: no puc moure'm amb seguretat.")
            self.stop_robot()
            return

        # segons el gest actual, cridem al mètode corresponent
        if self.current_action == GEST_ENDAVANT:
            self.action_explora_endavant()
        elif self.current_action == GEST_PATRULLA:
            self.action_patrulla()
        elif self.current_action == GEST_PARET:
            self.action_segueix_paret()
        else:
            self.get_logger().warn(f"Gest {self.current_action} no té acció associada.")
            self.finish_action(success=False)

    # ------------------------------------------------------------------
    # ACCIÓ 1: EXPLORACIÓ ENDAVANT SEGURA (GEST_ENDAVANT)
    # ------------------------------------------------------------------
    def action_explora_endavant(self):
        """
        Avança endavant mentre hi ha espai suficient davant.
        Pararem si:
          - ens acostem massa a un obstacle,
          - o s'arriba a una durada màxima de l'acció.
        """
        front = self.get_front_min_distance()
        if front is None:
            self.stop_robot()
            return

        if front < self.safe_front_dist:
            self.get_logger().info("Obstacle detectat endavant, aturant exploració.")
            self.finish_action(success=True)
            return

        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ------------------------------------------------------------------
    # ACCIÓ 3: MINI PATRULLA LOCAL (GEST_PATRULLA)
    # ------------------------------------------------------------------
    def action_patrulla(self):
        """
        Petita ruta en segments (endavant + girs).
        Exemple simple basat en temps, sempre vigilant obstacles.
        """
        front = self.get_front_min_distance()
        if front is None:
            self.stop_robot()
            return

        if front < self.safe_front_dist:
            self.get_logger().warn("Obstacle massa a prop durant la patrulla.")
            self.finish_action(success=False)
            return

        elapsed = time.time() - self.action_start_time
        twist = Twist()

        # seqüència simple: endavant -> gir esquerra -> endavant -> gir esquerra
        if self.patrol_segment_index == 0:
            if elapsed < 2.0:
                twist.linear.x = 0.15
                twist.angular.z = 0.0
            else:
                self.patrol_segment_index = 1
        if self.patrol_segment_index == 1:
            if elapsed < 3.0:
                twist.linear.x = 0.0
                twist.angular.z = 0.5
            else:
                self.patrol_segment_index = 2
        if self.patrol_segment_index == 2:
            if elapsed < 5.0:
                twist.linear.x = 0.15
                twist.angular.z = 0.0
            else:
                self.patrol_segment_index = 3
        if self.patrol_segment_index == 3:
            if elapsed < 6.0:
                twist.linear.x = 0.0
                twist.angular.z = 0.5
            else:
                self.finish_action(success=True)
                return

        self.cmd_pub.publish(twist)

    # ------------------------------------------------------------------
    # ACCIÓ 2: SEGUIR PARET (GEST_PARET)
    # ------------------------------------------------------------------
    def action_segueix_paret(self):
        """
        Segueix una paret al lateral dret de manera simple:
          - intenta mantenir una distància target_wall_dist a la paret dreta.
          - si no detecta paret, es desplaça lleugerament cap a la dreta
            per "buscar-la".
          - si hi ha obstacle de front massa a prop, gira una mica per esquivar-lo.
        L'acció dura un temps limitat.
        """
        if self.wall_follow_start_time is None:
            self.wall_follow_start_time = time.time()

        elapsed = time.time() - self.wall_follow_start_time
        if elapsed > self.wall_follow_duration:
            self.get_logger().info("Fi de seguiment de paret per temps.")
            self.finish_action(success=True)
            return

        front = self.get_front_min_distance()
        dist_right = self.get_side_distance(side="right")

        twist = Twist()
        base_speed = 0.12

        # obstacle frontal massa a prop -> girar una mica a l'esquerra
        if front is not None and front < self.safe_front_dist:
            twist.linear.x = 0.0
            twist.angular.z = 0.5
            self.cmd_pub.publish(twist)
            return

        # si no detectem paret al costat dret, desplaçar-nos per buscar-la
        if dist_right is None or dist_right > self.max_wall_dist:
            twist.linear.x = base_speed
            twist.angular.z = -0.25  # lleuger gir cap a la dreta
            self.cmd_pub.publish(twist)
            return

        # controlador proporcional per mantenir distància a la paret
        error = dist_right - self.target_wall_dist  # positiu si massa lluny
        k_p = 1.0
        twist.linear.x = base_speed
        twist.angular.z = -k_p * error  # si massa lluny, girem cap a la dreta

        # límit de gir
        twist.angular.z = max(min(twist.angular.z, 0.8), -0.8)

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
