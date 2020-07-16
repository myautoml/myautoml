# MyAutoML

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
cookiecutter https://github.com/erikjandevries/myautoml.git --directory="cookiecutter/binary_classifier"
```
