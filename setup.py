# Setup.py script used in Python projects to package and distribute the code. It describes the setup configuration for the project,
# Including dependencies, and makes it easier for others to install and use the project. 

#creating function for get_requirements
from setuptools import setup, find_packages
from typing import List

# By specifying the dependencies in the requirements.txt file, which is read by the get_requirements function,
# you ensure that anyone who installs your package also installs the necessary libraries.
# Users can install your package using pip by pointing to the setup.py file, which will automatically handle dependencies and installation.
def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements,
    excluding any editable installs like '-e .'
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if not req.startswith("-e")]
    return requirements

setup(
    name='ML-Project',
    version='0.0.1',
    author='Sanjay',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)