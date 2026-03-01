from setuptools import setup, find_packages

setup(
    name='mnpbem',
    version='0.1.0',
    description='Metallic Nanoparticle Boundary Element Method',
    packages=find_packages(),
    package_data={
        'mnpbem': ['materials/data/*'],
    },
    python_requires='>=3.11',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'matplotlib>=3.6.0',
    ],
)
