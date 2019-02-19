from setuptools import setup, find_packages


"""
Instructions for creating a release of the scispacy library.

1. Make sure your working directory is clean.
2. Make sure that you have changed the versions in "scispacy/version.py".
3. Create the distribution by running "python setup.py sdist" in the root of the repository.
4. Check you can install the new distribution in a clean environment.
5. Upload the distribution to pypi by running "twine upload <path to the distribution> -u <username> -p <password>".
   This step will ask you for a username and password - the username is "scispacy" you can
   get the password from LastPass.
"""

VERSION = {}
# version.py defines VERSION and VERSION_SHORT variables.
# We use exec here to read it so that we don't import scispacy
# whilst setting up the package.
with open("scispacy/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name = 'scispacy',
    version = VERSION["VERSION"],
    url = 'https://allenai.github.io/SciSpaCy/',
    author = 'Allen Institute for Artificial Intelligence',
    author_email = 'ai2-info@allenai.org',
    description = 'A full SpaCy pipeline and models for scientific/biomedical documents.',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords = ["bioinformatics nlp spacy SpaCy biomedical"],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license="Apache",
    install_requires=[
        "spacy==2.0.18",
        "awscli",
        "conllu",
        "numpy"
        ],
    tests_require=[
        "pytest",
        "pytest-cov",
        "pylint",
        "mypy"
        ],
    python_requires='>=3.5.0',
)
