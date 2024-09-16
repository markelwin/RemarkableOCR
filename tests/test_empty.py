# GitHub requires tests to pass workflow integrations and screen upgrades to pip package pushes.
# https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python
# https://stackoverflow.com/questions/75407052/installing-test-files-with-pyproject-toml-and-setuptools
# https://github.com/pydantic/pytest-examples/blob/main/pyproject.toml
# https://ianhopkinson.org.uk/2022/02/understanding-setup-py-setup-cfg-and-pyproject-toml-in-python/
#
# The test below is what we use for our default empty test to pass GitHub push checks.
# Note the pypi tutorial pyproject.toml needs the following line to find the test.
# https://packaging.python.org/en/latest/tutorials/packaging-projects/
# https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
#
# [tool.pytest.ini_options]
# testpaths = ["tests"]
#
# other useful notes to consider for pypi
# what to do if you botch a release: https://snarky.ca/what-to-do-when-you-botch-a-release-on-pypi/
# where the pyproject.toml standard came from: https://snarky.ca/what-the-heck-is-pyproject-toml/

import unittest


class TestEmpty(unittest.TestCase):

    def test_add_one(self):
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()