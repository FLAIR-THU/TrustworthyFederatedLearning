import os

from setuptools import find_packages, setup


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join(".", "requirements.txt")
    with open(reqs_path, "r") as f:
        requirements = [line.rstrip() for line in f]
    return requirements


__version__ = "0.0.0"
ext_modules = []

console_scripts = []

setup(
    name="ptbi",
    version=__version__,
    install_requires=read_requirements(),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    entry_points={"console_scripts": console_scripts},
    zip_safe=False,
)
