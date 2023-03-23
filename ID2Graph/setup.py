import os

from setuptools import find_packages, setup


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join(".", "requirements.txt")
    with open(reqs_path, "r") as f:
        requirements = [line.rstrip() for line in f]
    return requirements


console_scripts = []

setup(
    name="llatvfl",
    version="0.0.0",
    install_requires=read_requirements(),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={"console_scripts": console_scripts},
)
