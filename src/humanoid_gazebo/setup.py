"""Install the Gazebo Classic backend assets and launch file."""

from glob import glob
import os

from setuptools import setup

package_name = 'humanoid_gazebo'
data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]
for folder in ('launch', 'config', 'worlds', 'urdf', 'urdf/transmissions'):
    files = [path for path in glob(folder + '/*') if os.path.isfile(path)]
    data_files.append(('share/' + package_name + '/' + folder, files))

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bjarne',
    maintainer_email='grotelueschen@campus.tu-berlin.de',
    description='Gazebo Classic integration for the humanoid biped.',
    license='Apache-2.0',
    tests_require=['pytest'],
)
