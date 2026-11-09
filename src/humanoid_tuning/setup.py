# Copyright 2026 Bjarne
# SPDX-License-Identifier: Apache-2.0
"""Install the joint tuning panel and isolated simulator launch."""

from setuptools import setup


setup(
    name='humanoid_tuning',
    version='0.0.0',
    packages=['humanoid_tuning'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/humanoid_tuning']),
        ('share/humanoid_tuning', ['package.xml', 'plugin.xml']),
        ('share/humanoid_tuning/launch', ['launch/tuning.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bjarne',
    maintainer_email='grotelueschen@campus.tu-berlin.de',
    description='Floating joint PD tuning with rqt and PlotJuggler 4.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'tuning_panel = humanoid_tuning.panel:main',
    ]},
)
