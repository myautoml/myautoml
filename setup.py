from setuptools import setup, find_packages

from myautoml.version import VERSION

with open("README.md", "r") as fh:
    long_description = fh.read()

# Optional dependencies
extras = {
    'evaluation': ['shap'],
    'optimisation': ['hyperopt'],
    'tracking': ['mlflow'],
    'visualisation': ['matplotlib', 'seaborn']
}
# Meta dependency groups
extras['all'] = [item for group in extras.values() for item in group]

setup(
    name='myautoml',
    version=VERSION,
    description='My Auto ML - tools for running Data Science projects',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Erik Jan de Vries',
    author_email='erikjandevries@users.noreply.github.com',
    url='https://github.com/erikjandevries/myautoml',
    packages=find_packages(),
    install_requires=[
        'joblib',
        'numpy',
        'pandas',
        'pyyaml',
        'scikit-learn',
    ],
    extras_require=extras,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: None :: To be determined",
        "Operating System :: OS Independent",
    ],
)
