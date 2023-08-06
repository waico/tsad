__version__ = "1.0.2"

from setuptools import setup, find_packages
 
setup(name='tsad',
      version='1.0.0',
      url='https://github.com/waico/tsad',
      license='Attribution-NonCommercial-NoDerivatives 4.0 International',
      packages=find_packages(),
      #packages=['tsad'],
      author='Vyacheslav Kozitsin, Oleg Berezin, Iurii Katser, Ivan Maximov',
      author_email='rfptk2525@yandex.ru',
      description='Time Series Analysis for Simulation of Technological Processes',
      #packages=find_packages(exclude=['tests']),
      long_description=open('./tsad/README.md').read(),
      #long_description_content_type='text/x-rst',
      install_requires=['numpy>=1.19.5','pandas>=1.0.1','matplotlib>=3.1.3','scikit-learn>=0.24.1','torch>=1.10.0'], # install_requires=[ 'A>=1,<2', 'B>=2']
      zip_safe=False)
