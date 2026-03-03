import pytest
import field_axioms
import field_finite

def test_finite_field():
    field_axioms.check_field_axioms(field_finite)
