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
      include_package_data=True,
      install_requires=[
          'numpy',
          'pytest',
          'pandas',
          'scikit-image',
          'scikit-learn',
          'nibabel',
          'vtk',
          'dipy',
          'pyradiomics',
      ],
      zip_safe=False,
      classifiers=[
          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent',
          'Development Status :: 5 - Production/Stable'
      ], )
