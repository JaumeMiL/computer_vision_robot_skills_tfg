from setuptools import setup

package_name = 'gesture_recognition_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'opencv-python', 'mediapipe', 'numpy'],
    zip_safe=True,
    maintainer='jaumemil',
    maintainer_email='jaumemil@todo.todo',
    description='MediaPipe hands -> /gesture_command',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = gesture_recognition_pkg.vision_node:main',
        ],
    },
)
