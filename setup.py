'''
Author:
Email: 
Date: 2023-01-25 19:36:50
LastEditTime: 2023-01-25 19:44:06
Description: 
'''

from setuptools import setup, find_packages

setup(name='Safebench',
      packages=["gym_carla", "planning"],
      include_package_data=True,
      version='1.0.0',
      install_requires=['gym', 'pygame'])
