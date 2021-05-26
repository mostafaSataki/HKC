from setuptools import setup
setup(name='HKC',
version='1.0.1',
description='7khatcode Utility for public and machine learning project in python',
url='https://github.com/mostafaSataki/HKC',
author='Mostafa Sataki',
author_email='sataki.mostafa@email.com',
license='MIT',
packages=['HKC'],
install_requires=[
    'selenium',
    'object-detection'
    ],
zip_safe=False)