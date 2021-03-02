# Contributing to this repo

## Getting started
    It's strongly advised to create a virtual environment just for
    this repo. If you use
    [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/),
    it's as easy as `mkvirtualenv methopt && workon methopt`.

## Style guide
    We use [black](https://github.com/psf/black) — "an uncompromising
    code formatter" and the style it implies. Please, be sure to
    install it and use it with your favourite code editor.

## Project layout
    All code implementing optimisation methods goes to <file:src/optmeth/>.

    All new functionality should be covered by tests which go to <file:tests>.

## Installing
    To install the project (which includes building dependencies) run
    `python -m pip install .`.

## Testing
    Once you've installed run `pytest`.

    Please, note when adding new tests that new test files should end
    with `_test`. All function that are supposed to test something
    should be prefixed with "test_". It is one of the requirements of
    a [pytest](https://docs.pytest.org/en/latest/) test runner.
