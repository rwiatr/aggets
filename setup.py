from setuptools import setup, find_packages

setup(
    name='aggets',
    version='0.0.1',
    description='Aggregate Encoders',
    # long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rwiatr/agge',
    author='Roman Wiatr',
    author_email='author@example.com',
    keywords='aggregate encoders, logistic regression, CTR',


    package_dir={'': 'src'},
    packages=find_packages(where='src'),  # Required
    python_requires='=3.8',

    install_requires=['numpy', 'pandas', 'scikit-learn', 'scikitplot', 'pytorch'],
    # extras_require={
    #     'dev': [],
    #     'test': [],
    # },

    project_urls={
        'Paper': 'TBA',
        'Source': 'https://github.com/rwiatr/agge/',
    },
)