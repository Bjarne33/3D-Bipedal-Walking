"""Package setup for humanoid_control."""

from setuptools import find_packages, setup

package_name = 'humanoid_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/gait.yaml', 'config/estimator.yaml']),
        ('share/' + package_name + '/launch', ['launch/walk.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bjarne',
    maintainer_email='grotelueschen@campus.tu-berlin.de',
    description='Startup, test motions and model-based walking for the humanoid biped.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'humanoid_startup = humanoid_control.nodes.humanoid_startup:main',
            'com_visualizer = humanoid_control.nodes.com_visualizer:main',
            'humanoid_predefined_trajs = humanoid_control.nodes.humanoid_predefined_trajs:main',
            'humanoid_walk = humanoid_control.nodes.humanoid_walk:main',
            'humanoid_estimator = humanoid_control.nodes.humanoid_estimator:main',
        ],
    },
)
