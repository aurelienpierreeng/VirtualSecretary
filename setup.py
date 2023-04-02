from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Digital office framework'
LONG_DESCRIPTION = 'Digital office framework'

# Setting up
# Call python setup.py sdist after
setup(
       # the name must match the folder name 'verysimplemodule'
        name="virtualsecretary",
        version=VERSION,
        author="Aurélien Pierre",
        author_email="<contact@aurelienpierre.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that
        # needs to be installed along with your package. Eg: 'caer'

        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Office",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Linux :: Linux",
        ]
)
