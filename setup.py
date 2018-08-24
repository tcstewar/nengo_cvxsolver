from setuptools import setup

setup(
    name='nengo_cvxsolver',
    packages=['nengo_cvxsolver'],
    version='0.0.1',
    author='Terry Stewart',
    description='Decoder solver for nengo that uses cvxpy',
    install_requires=["cvxpy"],
    )
