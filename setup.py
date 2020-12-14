from pathlib import Path
from setuptools import setup, find_packages
import sys

base_dir = Path(__file__).parent
src_dir = base_dir / "src"

# When executing the setup.py, we need to be able to import ourselves, this
# means that we need to add the src/ directory to the sys.path.
sys.path.insert(0, src_dir)

about = {}
with open(src_dir / "myautoml" / "__about__.py") as f:
    exec(f.read(), about)

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
    name=about["__title__"],
    version=about["__version__"],

    description=about["__summary__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about["__uri__"],

    author=about["__author__"],

    project_urls={
        'Documentation': 'https://myautoml.readthedocs.io',
        "Source Code": "https://github.com/myautoml/myautoml",
    },

    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        # "License :: None :: To be determined",
        "Operating System :: OS Independent",
    ],

    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'click',
        'joblib',
        'mlflow',
        'numpy',
        'pandas',
        'pyyaml',
        'requests',
        'scikit-learn',

        'python-box',
        'python-dotenv',
        'envyaml',
    ],
    include_package_data=True,

    python_requires='>=3.6,<3.9',
    extras_require=extras,

    entry_points={
        "console_scripts": [
            "registermodel = myautoml.cli:registermodel"
        ]
    },
)
