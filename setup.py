from setuptools import find_packages, setup

setup(
    name='neat',
    packages=find_packages(include=['neat', 'neat/iznn', 'neat/nn', 'neat/ctrnn']),
    version='0.1.0',
    description='NEAT-python with tensorboard, origin: https://github.com/CodeReclaimers/neat-python',
    author='Oliver Halaš',
)