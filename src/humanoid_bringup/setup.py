"""Install the humanoid system-level launch files."""

from setuptools import find_packages, setup

package_name = 'humanoid_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/spawn_humanoid.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bjarne',
    maintainer_email='grotelueschen@campus.tu-berlin.de',
    description='System-level simulation launch for the humanoid biped.',
    license='Apache-2.0',
    tests_require=['pytest'],
)
