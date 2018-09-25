import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../SciSpaCy")

from dummy_test_coverage import dummy_test_coverage

def test_testing():
    dummy_test_coverage()
    test_variable_1 = 1
    test_variable_2 = 1
    assert (test_variable_1 + test_variable_2) == 2
