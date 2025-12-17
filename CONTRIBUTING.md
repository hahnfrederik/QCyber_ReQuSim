# Contributing

Code Contributions to requsim should be made on on their own Git
branches via a pull request to its `dev` branch. The pull request will
first be reviewed and merged to `dev` and, after further evaluation,
propagated to the `main` branch from where it can be included into
an official release.


## Development Environment

This project uses Python 3.12 and virtualenv for setting up the development
environment:
```
git clone https://github.com/jwallnoefer/requsim.git
cd requsim
git checkout dev
python3.12 -m venv .venv
source .venv/bin/activate
```
When the virtual environment is active, you can then install all the exact
versions of this project's dependencies and set up pre-commit like this:
```
pip install -r requirements.txt
pre-commit install
```


## Tests

It is strongly encouraged to write tests and run them locally prior to any
pull request to the GitHub repository.
Tests are located in the `./tests/` directory and we use
[`pytest`](https://docs.pytest.org/en/stable/) as our testrunner. (Note that
some old tests are still written as `unittest` test cases.)

You can invoke `pytest` in the development environment by simply running:
```
pytest
```
This should automatically discover all the tests and run in under 10 seconds.
