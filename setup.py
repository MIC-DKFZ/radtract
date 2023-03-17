from setuptools import setup

setup(name='radtract',
      version='0.1.0',
      description='',
      long_description='',
      url='https://github.com/MIC-DKFZ/radtract/',
      author='Peter F. Neher',
      author_email='p.neher@dkfz.de',
      license='Apache 2.0',
      packages=['radtract'],
      install_requires=[
          'dipy',
          'pyradiomics',
          'scikit-image',
          'scikit-learn',
          'vtk',
          'pandas',
          'nibabel',
          'numpy'
      ],
      zip_safe=False,
      classifiers=[
          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent',
          'Development Status :: 5 - Production/Stable'
      ], )
