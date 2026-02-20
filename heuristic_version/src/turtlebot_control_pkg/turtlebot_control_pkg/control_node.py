#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Node de Control (Versió Heurística) per al Robot TurtleBot.

Aquest node s'encarrega d'interpretar els gestos provinents de visió i 
generar patrons de moviment basats en heurístiques (explorar, patrullar o 
seguir paret). Compta amb un bucle de seguretat reactiu que atura el 
robot si el làser (LaserScan) detecta col·lisions imminents.
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int8

# ==========================================
#         DEFINICIÓ DE GESTOS I ORDRES
# ==========================================
# Codis de gest (coherents amb vision_node.py)
GEST_PARAR      = 0   # puny
GEST_ENDAVANT   = 1   # mà oberta -> exploració endavant segura
GEST_PARET      = 2   # polze amunt -> seguir paret
GEST_PATRULLA   = 3   # dos dits -> mini patrulla
GEST_REPETIR    = 10  # cercle -> repetir última acció

# Convencions senzilles per la integració amb les ordres de braç
ARM_NEUTRE = 0
ARM_OK     = 1
ARM_FAIL   = -1


class ControlNode(Node):
    """
    Node de control d'alt nivell:
    - Rep gestos (Int8) pel tòpic `/gesture_command`.
    - Segons el gest, inicia una acció autònoma (explorar, patrullar, seguir paret...).
    - Supervisor actiu del LaserScan temporal per evitar xocs.
    - Notifica d'èxits o fallides publicant al tòpic `/arm_command`.
    """

    def __init__(self):
        super().__init__('control_node')

        # ==========================================
        #         SUBSCRIPTORS I PUBLICADORS
        # ==========================================
        self.subscription = self.create_subscription(
            Int8,
            '/gesture_command',
            self.gesture_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_pub = self.create_publisher(Int8, '/arm_command', 10)

        # ==========================================
        #         ESTAT DEL LÀSER
        # ==========================================
        self.last_scan = None
        self.last_scan_time = self.get_clock().now()

        # ==========================================
        #         FSM D'ACCIÓ
        # ==========================================
        self.current_action = None          # Codi de gest que s'està executant
        self.action_start_time = None
        self.action_state = None            # Sub-estat intern
        self.last_successful_action = None  # Caxé per permetre al GEST_REPETIR actuar
        
        self.safe_front_dist = 0.4
        self.max_action_duration = 8.0      # Durada màxima de la majoria d'ordres (s)

        # Variables patrulla
        self.patrol_segment_index = 0

        # Variables wall-follow (seguir paret)
        self.wall_follow_start_time = None
        self.wall_follow_duration = 6.0     # Segons d'exploració de parets
        self.target_wall_dist = 0.6
        self.max_wall_dist = 1.5

        # Bucle de decisió (a 10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Node de Control (Heuristic) iniciat i llest.')

    # ==================================================================
    #         CALLBACKS
    # ==================================================================
    def scan_callback(self, msg: LaserScan):
        """Callback que emmagatzema periòdicament el resultat de lidar scan."""
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()

    def gesture_callback(self, msg: Int8):
        """Processador dels esdeveniments capturats per visió i assignador d'ordres."""
        gest = msg.data
        self.get_logger().info(f'Rebut gest: {gest}')

        # GEST_PARAR: cancel·lar tota acció de manera immediata (Estabilització)
        if gest == GEST_PARAR:
            self.stop_robot()
            self.finish_action(success=True)  # Donem per aturada "correcta"
            return

        # Gest dinàmic: Repetir l'última acció vàlida enregistrada
        if gest == GEST_REPETIR:
            if self.last_successful_action is not None:
                self.get_logger().info(f'Repetint acció {self.last_successful_action}')
                self.start_action(self.last_successful_action)
            else:
                self.get_logger().warn('No hi ha cap acció prèvia per repetir.')
            return

        # Subsitueix la tasca actual (Overriding) per la nova comanda exigida.
        self.start_action(gest)

    # ==================================================================
    #         GESTIÓ D'ACCIONS
    # ==================================================================
    def start_action(self, gest: int):
        """Reseteja estats temporals i engega la tasca demanada."""
        self.get_logger().info(f'Iniciant acció per al gest {gest}')
        self.current_action = gest
        self.action_start_time = time.time()
        self.action_state = 0

        if gest == GEST_PATRULLA:
            self.patrol_segment_index = 0
        elif gest == GEST_PARET:
            self.wall_follow_start_time = time.time()

        # Situa el braç en una posició estàtica i neutra en iniciar feina
        self.send_arm_command(ARM_NEUTRE)

    def finish_action(self, success: bool):
        """Marca una acció com acabada, atura les rodes i interacciona amb el braç."""
        if self.current_action is None:
            return

        self.stop_robot()

        if success:
            self.get_logger().info(f"Acció {self.current_action} completada amb èxit.")
            self.last_successful_action = self.current_action
            
            # Notifica que s'ha completat l'acció
            self.send_arm_command(ARM_OK)   
        else:
            self.get_logger().warn(f"Acció {self.current_action} NO s'ha pogut completar.")
            
            # Gest de disconformitat
            self.send_arm_command(ARM_FAIL) 

        # Restaura estat idle
        self.current_action = None
        self.action_start_time = None
        self.action_state = None

    # ==================================================================
    #         UTILITATS BÀSIQUES DE NAVEGACIÓ
    # ==================================================================
    def stop_robot(self):
        """Llença velocitat (0,0) per a forçar un stop cinemàtic."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def send_arm_command(self, state: int):
        """Converteix el state integer i publica cap on dirigir la postura del braç."""
        msg = Int8()
        msg.data = state
        self.arm_pub.publish(msg)

    def get_front_min_distance(self):
        """Processa el vector de ranges per trobar la mínima distància frontal."""
        if self.last_scan is None:
            return None

        ranges = np.array(self.last_scan.ranges, dtype=float)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        angle_min = self.last_scan.angle_min
        angle_inc = self.last_scan.angle_increment
        angles = angle_min + np.arange(len(ranges)) * angle_inc

        # Sector frontal restringit +- 20 graus
        mask_front = np.logical_and(angles > -math.radians(20),
                                    angles <  math.radians(20))

        front_ranges = ranges[mask_front]
        if front_ranges.size == 0:
            return None
            
        return float(np.min(front_ranges))

    def get_side_distance(self, side: str = "right"):
        """Permet consultar un sector angular per a una lectura dreta/esquerra directa."""
        if self.last_scan is None:
            return None

        ranges = np.array(self.last_scan.ranges, dtype=float)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        angle_min = self.last_scan.angle_min
        angle_inc = self.last_scan.angle_increment
        angles = angle_min + np.arange(len(ranges)) * angle_inc

        if side == "right":
            # Sector lateral dret (-120º a -60º)
            mask = np.logical_and(angles < -math.radians(60),
                                  angles > -math.radians(120))
        else:
            # Sector lateral esquerra (60º a 120º)
            mask = np.logical_and(angles > math.radians(60),
                                  angles < math.radians(120))

        side_ranges = ranges[mask]
        if side_ranges.size == 0:
            return None
            
        return float(np.min(side_ranges))

    # ==================================================================
    #         BUCLE PRINCIPAL DE CONTROL ESTATAL
    # ==================================================================
    def control_loop(self):
        """Funciona a una freqüència prefixada per anar monitoritzant la progressió."""
        # Standby, esperant que rebi un mode actiu.
        if self.current_action is None:
            return

        # Seguretat global: Abortar si porta actiu més temps que el limit de seguretat.
        if self.action_start_time is not None:
            if time.time() - self.action_start_time > self.max_action_duration:
                self.get_logger().warn("Temps màxim d'acció excedit, aturant per seguretat.")
                self.finish_action(success=False)
                return

        # Sense dades làser fiables recents, s'atura en sec.
        if self.last_scan is None:
            self.get_logger().warn_throttle(2.0, "Sense LaserScan recent: impossible moure's amb visibilitat.")
            self.stop_robot()
            return

        # Execució de polimorfismes específics d'acció en base a el flag d'acció
        if self.current_action == GEST_ENDAVANT:
            self.action_explora_endavant()
        elif self.current_action == GEST_PATRULLA:
            self.action_patrulla()
        elif self.current_action == GEST_PARET:
            self.action_segueix_paret()
        else:
            self.get_logger().warn(f"Gest {self.current_action} no està suportat pel routing actual.")
            self.finish_action(success=False)

    # ------------------------------------------------------------------
    # ACCIÓ 1: EXPLORACIÓ ENDAVANT SEGURA (GEST_ENDAVANT)
    # ------------------------------------------------------------------
    def action_explora_endavant(self):
        """
        Avança endavant mentre hi ha espai suficient al davant.
        Pararem si ens acostem massa a un obstacle, o si expira la durada temporal.
        """
        front = self.get_front_min_distance()
        if front is None:
            self.stop_robot()
            return

        if front < self.safe_front_dist:
            self.get_logger().info("Obstacle detectat davant! Aturant mode d'exploració.")
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
        Petita ruta planificada seqüencialment.
        Combina un endavant ràpid i un gir d'escombrat tot vigilant els obstacles.
        """
        front = self.get_front_min_distance()
        if front is None:
            self.stop_robot()
            return

        if front < self.safe_front_dist:
            self.get_logger().warn("Paret massa a prop durant patrulla. Fallada planificada.")
            self.finish_action(success=False)
            return

        elapsed = time.time() - self.action_start_time
        twist = Twist()

        # Seqüència temporal simple: Avanç -> Gir Esquerra -> Avanç -> Gir Esquerra
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
        Segueix una paret lateral dreta de costat (Simple Controlador Proporcional):
          - Intenta mantenir una target_wall_dist desitjada cap a la paret detectada.
          - Si no hi ha paret (espai buit al lateral dret), s'arrambarà diagonalment.
          - Si el sensor frontal marca bloqueig, pivotarà a esquerra per esquivar recursivament.
        """
        if self.wall_follow_start_time is None:
            self.wall_follow_start_time = time.time()

        elapsed = time.time() - self.wall_follow_start_time
        
        # Superació del temps teòric delimitat i acaba en mode correcte (Success=True)
        if elapsed > self.wall_follow_duration:
            self.get_logger().info("Esgotat el temps estimat pel seguiment de paret completat satisfactòriament.")
            self.finish_action(success=True)
            return

        front = self.get_front_min_distance()
        dist_right = self.get_side_distance(side="right")

        twist = Twist()
        base_speed = 0.12

        # Risc de xoc frontal. S'evita girant una mica cap a l'esquerra fins netejar el costat fosc.
        if front is not None and front < self.safe_front_dist:
            twist.linear.x = 0.0
            twist.angular.z = 0.5
            self.cmd_pub.publish(twist)
            return

        # Falta parets o estàs massa en obert. Deriva natural cap a on teóricament s'esperaria el mur.
        if dist_right is None or dist_right > self.max_wall_dist:
            twist.linear.x = base_speed
            twist.angular.z = -0.25  # Lleuger gir cap a la dreta buscant la paret
            self.cmd_pub.publish(twist)
            return

        # Controlador 'P'
        error = dist_right - self.target_wall_dist  # (Positiu si estic més lluny del target)
        k_p = 1.0
        
        twist.linear.x = base_speed
        twist.angular.z = -k_p * error  # Quan més error positiu hi ha, més cap en "-k_p" (dreta direccional z es negativa) l'estirem.

        # Control del clamping sobre la velocitat angular
        twist.angular.z = max(min(twist.angular.z, 0.8), -0.8)

        self.cmd_pub.publish(twist)


def main(args=None):
    """
    Funció principal d'arrencada amb prevenció d'excepcions on es bloqueja 
    la velocitat a valors parats abans del reciclatge Node.
    """
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
