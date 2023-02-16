from setuptools import find_packages, setup

setup(
    name='gymnasium-pomdps',
    version='1.0.0',
    description='Gym flat POMDP environments',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/gym-pomdps',
    packages=find_packages(include=['gymnasium_pomdps', 'gymnasium_pomdps.*']),
    package_data={'': ['*.pomdp']},
    install_requires=[
        'gymnasium',
        'numpy',
        'one_to_one',
        'rl_parsers',
    ],
    license='MIT',
)
