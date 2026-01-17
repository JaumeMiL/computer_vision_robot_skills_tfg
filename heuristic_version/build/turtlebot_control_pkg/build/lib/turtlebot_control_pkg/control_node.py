import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
from geometry_msgs.msg import Twist

class ControlNode(Node):
    """
    Aquest node es subscriu al tòpic /gesture_command.
    Quan rep un missatge, el tradueix a una comanda de velocitat (Twist)
    i la publica al tòpic /cmd_vel per moure el robot.
    """
    def __init__(self):
        super().__init__('control_node')
        
        # 1. Creador del subscriptor (Subscriber)
        # Escolta missatges Int8 al tòpic /gesture_command i crida la funció gesture_callback
        self.subscription = self.create_subscription(
            Int8,
            '/gesture_command',
            self.gesture_callback,
            10)
        
        # 2. Creador del publicador (Publisher)
        # Publicarà missatges Twist al tòpic /cmd_vel
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # 3. Mapeig de comandes
        # Aquest diccionari tradueix el número del gest a una velocitat
        # Estructura: {gest: (velocitat_lineal_x, velocitat_angular_z)}
        self.gesture_map = {
            0: (0.0, 0.0),    # 0: Puny Tancat -> Aturar
            1: (0.2, 0.0),    # 1: Palma Oberta -> Endavant
            2: (-0.2, 0.0),   # 2: Polze Amunt -> Enrere
            3: (0.0, 0.5),    # 3: Gira Esquerra (dos dits)
            4: (0.0, -0.5),   # 4: Gira Dreta (un dit)
        }
        
        self.get_logger().info('Node de Control iniciat i llest.')

    def gesture_callback(self, msg):
        """
        Aquesta funció s'executa cada cop que arriba un missatge de gest.
        """
        gesture_id = msg.data
        
        # Busquem la velocitat corresponent al gest rebut
        if gesture_id in self.gesture_map:
            linear_x, angular_z = self.gesture_map[gesture_id]
            
            # Creem el missatge de Twist
            twist_msg = Twist()
            twist_msg.linear.x = linear_x
            twist_msg.angular.z = angular_z
            
            # Publicar el missatge de moviment
            self.publisher_.publish(twist_msg)
            self.get_logger().info(f'Gest rebut: {gesture_id} -> Movent: lin_x={linear_x}, ang_z={angular_z}')
        else:
            self.get_logger().warn(f'Gest desconegut rebut: {gesture_id}')


def main(args=None):
    rclpy.init(args=args)
    control_node = ControlNode()
    try:
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        pass
    finally:
        control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
