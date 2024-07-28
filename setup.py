from setuptools import setup, find_packages

setup(
    name='HKC',
    version='1.0.1',
    description='7khatcode Utility for public and machine learning project in python',
    url='https://github.com/mostafaSataki/HKC',
    author='Mostafa Sataki',
    author_email='sataki.mostafa@email.com',
    license='MIT',
    packages=find_packages(),
    package_data={
        '': ['.\GT\*', '.\align\*', '.\models\*', '.\mtcnn\*', '.\open_vino\*',".\\util\*",".\VOC\*",".\Siamese\*"], 

    },
    include_package_data=True,
    install_requires=[
        'selenium',
    ],
    zip_safe=False
)