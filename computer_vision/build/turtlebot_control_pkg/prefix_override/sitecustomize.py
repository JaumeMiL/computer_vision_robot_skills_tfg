import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jaumemil/tfg_jaume_vf/install/turtlebot_control_pkg'
