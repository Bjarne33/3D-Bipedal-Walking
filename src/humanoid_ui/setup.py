# Copyright 2026 Bjarne
# SPDX-License-Identifier: Apache-2.0
"""Install the walk manager rqt panel."""

from setuptools import setup


setup(
    name='humanoid_ui',
    version='0.0.0',
    packages=['humanoid_ui'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/humanoid_ui']),
        ('share/humanoid_ui', ['package.xml', 'plugin.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bjarne',
    maintainer_email='grotelueschen@campus.tu-berlin.de',
    description='rqt panel to start and stop walking through the walk manager.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'walk_manager_panel = humanoid_ui.walk_manager_panel:main',
    ]},
)
