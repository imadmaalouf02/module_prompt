from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md') as f:
    long_description = f.read()

# Read the contents of the requirements file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='Prompt_Analyzer',
    version='0.1.0',
    author='GIIA_DS',
    author_email='your.email@example.com',
    description='A brief description of your project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SAAD1190/Prompt_Analyzer',
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: YOUR LICENSE HERE',  # Adjust this line
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
