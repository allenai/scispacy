from setuptools import setup, find_packages

setup(
    name = 'scispacy',
    version = '1.0.0',
    url = 'https://github.com/allenai/SciSpaCy',
    author = 'Allen AI2',
    author_email = 'ai2-info@allenai.org',
    description = 'SciSpaCy Package',
    packages = find_packages(),
    install_requires = ['spacy==2.0.12'],
)