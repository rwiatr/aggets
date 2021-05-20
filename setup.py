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
    python_requires='=3.6',

    install_requires=['numpy', 'pandas', 'scikit-learn', 'futures', 'matplotlib', 'torch==1.6.0', 'wget'],
    # conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
    tests_require=['pytest'],
    # extras_require={
    #     'dev': [],
    #     'test': [],
    # },

    project_urls={
        'Paper': 'TBA',
        'Source': 'https://github.com/rwiatr/agge/',
    },
)
