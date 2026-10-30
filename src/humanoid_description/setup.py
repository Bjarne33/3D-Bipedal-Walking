"""Package setup for humanoid_description."""

from setuptools import find_packages, setup
import os


def package_files(directory):
    """List all files below a directory.

    Args:
        directory: Directory to walk recursively.

    Returns:
        Paths of all files found.
    """
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/humanoid_description']),
    ('share/humanoid_description', ['package.xml']),
    ('share/humanoid_description/launch', ['launch/humanoid_rviz.launch.py']),
]

for folder in ['urdf', 'config', 'rviz', 'worlds']:
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
    description='Robot model, Gazebo world, controllers and launch files for the biped.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
