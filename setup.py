import setuptools

__version__ = '0.1.1'

with open('README.md', 'r') as f:
    long_desc = f.read()

setuptools.setup(
    name='bix_analysis_libraries',
    version=__version__,
    author='Viktor Skorjanc',
    author_email='viktor.skorjanc@gmail.com',
    description='An assortment of analysis libraries.',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/vskorjanc/analysis-libraries',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'bric_analysis_libraries==0.0.12',
        'pandas==1.4.1',
        'numpy==1.21.2',
        'matplotlib==3.4.3',
        'scipy==1.7.2',
        'plotly==5.11.0',
        'thot-data==0.6.3',
        'thot-cli==0.5.1',
        'tabulate==0.9.0',
        'BaselineRemoval==0.1.1'
    ]
)
