from setuptools import setup

setup(name='helpers',
      version='1.0',
      description='Selection of helper functions',
      author='SR',
      author_email='stephen.aj.rawson@gmail.com',
      license='MIT',
      packages=['helpers'],
      install_requires=[
          "matplotlib",
          "numpy",
          "pandas",
          "seaborn",
          "scipy",
          "sklearn",
      ],
      zip_safe=False)
