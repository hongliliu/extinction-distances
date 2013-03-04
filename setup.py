#!/usr/bin/env python

import sys
if 'build_sphinx' in sys.argv or 'develop' in sys.argv:
    from setuptools import setup, Command
else:
    from distutils.core import setup, Command

with open('README.md') as file:
    long_description = file.read()

#with open('CHANGES') as file:
#    long_description += file.read()

# no versions yet from extinction_distance import __version__ as version

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


setup(name='extinction_distance',
      version='0.1',
      description='Extinction-based Distance Measurements.',
      long_description=long_description,
      author='Jonathan Foster',
      author_email='jonathan.b.foster@yale.edu',
      url='https://github.com/jfoster17/extinction-distances',
      packages=['extinction_distance',
          'extinction_distance.support', 
          'extinction_distance.completeness', 
          'extinction_distance.distance',
          'extinction_distance.images'], 
      cmdclass = {'test': PyTest},
     )
