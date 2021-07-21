from setuptools import setup, find_packages

setup(
    name='fpoffline',
    version='0.1.0dev', # e.g. 0.1.9dev
    description='Offline focal-plane analysis support',
    url='http://github.com/desihub/fpoffline',
    author='David Kirkby',
    author_email='dkirkby@uci.edu',
    license='MIT',
    packages=find_packages(exclude=["tests",]),
    install_requires=['numpy', 'scipy', 'fitsio', 'pandas'],
    include_package_data=False,
    zip_safe=False,
    entry_points = {
        'console_scripts': [
        ],
    }
)
