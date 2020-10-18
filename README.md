# MyAutoML

[![Documentation Status](https://readthedocs.org/projects/myautoml/badge/?version=latest)](https://myautoml.readthedocs.io/en/latest/?badge=latest)
[![Latest Python Release](https://img.shields.io/pypi/v/myautoml.svg)](https://pypi.org/project/myautoml/)
[![Downloads](https://pepy.tech/badge/myautoml)](https://pepy.tech/project/myautoml)

MyAutoML is a project that aims to help data scientists become more efficient, by providing:

- Cookiecutter templates
- Example scripts (based on the Cookiecutter templates)
- A Python package with functions to perform common tasks
- A programming framework to automate as much as possible of the repetitive
  work a data scientist is likely to encounter.

### Getting started

Install MyAutoML using `pip`:
```shell script
pip install myautoml
```

Import the Python package:
```python
import myautoml as maml
print(maml.__version__)
```

Further documentation is under development.
For now, have a look at the example `scripts`.


### Cookiecutters

To use the Cookiecutter templates:
```shell script
cookiecutter https://github.com/myautoml/myautoml.git --directory="cookiecutter/binary_classifier"
```
