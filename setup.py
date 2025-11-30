import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'harrp'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*launch.py')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='demulab-kohei',
    maintainer_email='demulab-kohei@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'reid3d = {package_name}.reid3d:main',
            f'pub_pillar_bbox = {package_name}.pub_pillar_bbox:main',
            f'ground_removal = {package_name}.ground_removal:main',
        ],
    },
)
