import pytest
import field_axioms
from field_finite import FiniteField
from field_rational import RationalField

def test_finite_field():
    field_axioms.check_field_axioms(FiniteField(7))

def test_rational_field():
    field_axioms.check_field_axioms(RationalField())
