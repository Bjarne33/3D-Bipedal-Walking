from setuptools import find_packages, setup
from glob import glob
import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/humanoid_description']),
    ('share/humanoid_description', ['package.xml']),
]

for folder in ['urdf', 'launch', 'config', 'rviz', 'worlds']:
    files = package_files(folder)
    for file in files:
        install_path = os.path.join('share/humanoid_description', os.path.dirname(file))
        data_files.append((install_path, [file]))
        
package_name = 'humanoid_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bjarne',
    maintainer_email='grotelueschen@campus.tu-berlin.de',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
