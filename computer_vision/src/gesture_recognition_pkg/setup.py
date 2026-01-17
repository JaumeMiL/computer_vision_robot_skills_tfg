from setuptools import setup
import os
from glob import glob

package_name = 'gesture_recognition_pkg'

setup(
    name=package_name,
    version='0.0.4',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('gesture_recognition_pkg/models/*.pth')),
    ],
    install_requires=['setuptools', 'opencv-python', 'mediapipe', 'numpy', 'torch'],
    zip_safe=True,
    maintainer='jaumemil',
    maintainer_email='jaumemil@todo.todo',
    description='Eureka Gesture Recognition -> /gesture_command',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = gesture_recognition_pkg.vision_node:main',
        ],
    },
)