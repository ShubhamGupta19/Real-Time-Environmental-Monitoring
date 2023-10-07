from setuptools import setup, find_packages
from typing import List

def get_requirements(filename:str)->List[str]:
    with open(filename) as f:
        requirements = f.readlines()
        requirements= [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
        return requirements



setup(
    name="Realtime-Environment-Monitoting",
    version='0.0.1',
    author="Shubham Gupta",
    author_email="shubhamgupta2048@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )