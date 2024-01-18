from setuptools import setup, find_packages

setup(
    name='qplay',
    version='0.111',
    packages=find_packages(),
    description='Quantum Play Language Framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Metamer',
    author_email='michael@metamermusic.com',
    url='https://github.com/jenkinsm13/qplay',
    install_requires=[
    'numpy',
    'scipy',
    'matplotlib',
    'ipywidgets',
    'pandas',
    'sympy',
    'qiskit',
    'networkx',
    'requests',
    'cirq',
    'strawberryfields',
    'pyquil',
    'pytket',
    'pennylane',
    'circuitpython',
    ],
)
