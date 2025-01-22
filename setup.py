from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]

# Parse the requirements
requirements = parse_requirements('requirements.txt')

# Setup configuration
setup(
    name='llm-for-political-science', 
    version='0.1.0',
    author='Collin Coil',
    author_email='collin.a.coil@gmail.com',
    url='https://github.com/CollinCoil/llm-for-political-science',
    description='This repository contains code to accompany the paper "Large Language Models: A Survey with Applications in Political Science". All code and data to replicate the study can be accessed in this repository.',
    packages=find_packages(),  # Automatically find all packages in the directory
    install_requires=requirements,  # Load the requirements
    include_package_data=True, 
    zip_safe=False,
)
