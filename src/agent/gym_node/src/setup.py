from setuptools import setup, find_packages

setup(name='red_team',
      packages=["gym_carla", "planning"],
      include_package_data=True,
      version='0.0.1',
      install_requires=['gym', 'pygame'])
