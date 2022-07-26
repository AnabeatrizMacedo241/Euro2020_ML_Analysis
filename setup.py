from setuptools import setup

with open('README.md', encoding="utf-8") as f:
    long_description = f.read()

setup(
    name = 'Euro2020_API',
    version = '0.0.3',
    url = 'https://github.com/AnabeatrizMacedo241/NBA_analysis',
    description = 'Get insights from the Euro 2020 data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = 'Ana Beatriz Macedo',
    author_email = '<anabeatrizmacedo241@gmail.com>', 
    packages = ['Euro2020_API'],
    install_requires=[
        'pandas',
        'selenium'],
    license= 'MIT',
    keywords = ['API', 'football', 'euro', 'soccer', 'Sports'],
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Topic :: Database',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],)
