__version__ = "0.19.3"

from setuptools import setup, find_packages
 
setup(
    name='tsad',
    version=__version__,
    python_requires = '>=3.10.0',
    url='https://github.com/waico/tsad',
    license='GNU GPLv3',
    packages=find_packages(exclude=['tests']),
    author='Viacheslav Kozitsin, Oleg Berezin, Iurii Katser, Ivan Maximov',
    author_email='rfptk2525@yandex.ru',
    description='Time Series Analysis for Simulation of Technological Processes',
    long_description=open('./tsad/README.md').read(),
    install_requires=[
        'ipykernel==6.25.1',
        'ipython==8.14.0',
        'matplotlib==3.7.1',
        'missingno==0.5.2',
        'numpy==1.25.0',
        'pandas==1.5.3',
        'plotly==5.16.1',
        'plotly-resampler==0.9.1',
        'pyarrow==14.0.1',
        'scikit-learn==1.1.2',
        'tsfel==0.1.6',
        'tsflex==0.3.0',
        'tsfresh==0.20.1',
        'torch==1.11.0',
        'xlrd==2.0.1'
        ], # install_requires=[ 'A>=1,<2', 'B>=2']
    zip_safe=False)


