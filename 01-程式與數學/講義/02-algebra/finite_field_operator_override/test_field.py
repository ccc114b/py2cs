import pytest
import field_axioms
from field_finite import FiniteField

def test_finite_field():
    field_axioms.check_field_axioms(FiniteField(7))
