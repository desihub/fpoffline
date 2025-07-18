from setuptools import setup, find_packages

setup(
    name="fpoffline",
    version="0.1",  # e.g. 0.1dev
    description="Offline focal-plane analysis support",
    url="http://github.com/desihub/fpoffline",
    author="David Kirkby",
    author_email="dkirkby@uci.edu",
    license="MIT",
    packages=find_packages(
        exclude=[
            "tests",
        ]
    ),
    install_requires=["numpy", "scipy", "fitsio", "pandas"],
    include_package_data=True,
    package_data={
        "fpoffline.data": ["*.csv"]
    },
    zip_safe=False,
    entry_points = {
        'console_scripts': [
            'endofnight=fpoffline.scripts.endofnight:main',
            'update_offset_p=fpoffline.scripts.update_offset_p:main',
            'revert_offset_p=fpoffline.scripts.revert_offset_p:main',
        ],
    },
)
