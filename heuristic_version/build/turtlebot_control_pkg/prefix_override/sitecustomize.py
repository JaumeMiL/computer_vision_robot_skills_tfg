import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jaumemil/tfg_jaume_v2/install/turtlebot_control_pkg'
