# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

# import nltk

from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install

with open('requirements.txt', 'r') as f:
    setup_requires = f.readlines()


def install_prerequisites():
    pass


class Install(_install):
    def run(self):
        _install.do_egg_install(self)
        install_prerequisites()
        _install.run(self)


class Develop(_develop):
    def run(self):
        install_prerequisites()
        _develop.run(self)


setup(name='gntp',
      version='1.0',
      description='Greedy Neural Theorem Prover',
      author='Pasquale Minervini',
      author_email='p.minervini@cs.ucl.ac.uk',
      url='https://github.com/uclnlp/gntp',
      test_suite='tests',
      license='MIT',
      install_requires=setup_requires,
      extras_require={
            'tf': ['tensorflow>=1.4.0'],
            'tf_gpu': ['tensorflow-gpu>=1.4.0'],
      },
      setup_requires=setup_requires,
      tests_require=setup_requires,
      cmdclass={
            'install': Install,
            'develop': Develop
      },
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules'
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      packages=find_packages(),
      keywords='tensorflow machine learning')
