import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../SciSpaCy/")

from dummy_test_coverage import dummy_test_coverage

def test_dummy_test():
    dummy_test_coverage()
    variable_1 = 1
    variable_2 = 1
    assert variable_1 + variable_2 == 2