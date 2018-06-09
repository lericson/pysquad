from setuptools import setup

setup(name='squad',
      version='0.1.0',
      author='Ludvig Ericson',
      author_email='ludvig@lericson.se',
      description='Simulate Quads. Or other multicopters.',
      url='http://sendapatch.se/',
      package_dir={'squad': 'squad'},
      packages=['squad'],
      install_requires=['pyqtgraph', 'numba', 'numpy', 'scipy', 'logcolor'])
