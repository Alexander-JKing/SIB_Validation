from setuptools import setup, find_packages

setup(name='sib_main',
      version='0.0.1',
      description='Collection of functions to carry out SIB Validation on Object Detection Algorithm results.',
      author='Alex King',
      author_email='alex.king@icon.ie',
      packages=find_packages(include=['sib_main', 'sib_main.*']),
      install_requires=['pandas>=1.3.2',
                        'geopandas>=0.9.0',
                        'numpy>=1.21.2',
                        'shapely>=1.7.1'])

