from setuptools import setup

package_name = 'turtlebot_control_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jaumemil',
    maintainer_email='jaumemil@todo.todo',
    description='Gesture Int8 -> cmd_vel Twist',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = turtlebot_control_pkg.control_node:main',
        ],
    },
)
