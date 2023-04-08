from setuptools import setup,find_packages
# from typing import List


#Declaring variables for setup functions
PROJECT_NAME="Advanced House Price Prediction"
VERSION="0.0.1"
AUTHOR="Prathik Vishavadiya"
DESRCIPTION="Machine learning project to predict the sale price of a house"
REQUIREMENT_FILE_NAME="requirements.txt"




def get_requirements_list():
    """
    Description: This function is going to return list of requirement
    mention in requirements.txt file
    return This function is going to return a list which contain name
    of libraries mentioned in requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        print(requirement_list)
        if "-e ." in requirement_list:
            requirement_list.remove("-e .")
        return requirement_list


setup(
name=PROJECT_NAME,
version=VERSION,
author=AUTHOR,
description=DESRCIPTION,
packages=find_packages(),  # housing is a package(it has __init__)
install_requires=get_requirements_list() # to install extra libraries not in the packages
)
