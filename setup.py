import setuptools

with open( 'README.md', 'r' ) as f:
	long_desc = f.read()

setuptools.setup(
	name='bix_analysis_libraries',
	version = '0.0.3',
	author='Viktor Skorjanc',
	author_email = 'viktor.skorjanc@gmail.com',
	description = 'An assortment of analysis libraries.',
	long_description = long_desc,
	long_description_content_type = 'text/markdown',
	url = 'https://github.com/vskorjanc/analysis-libraries',
	packages = setuptools.find_packages(),
	classifiers = [
		"Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
	],
	install_requires = [
		'bric_analysis_libraries',
		'pandas',
		'numpy'
	]
)
