# test_my_math.py
from my_math import add

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-1, -2) == -3

def test_add_zero():
    assert add(0, 5) == 5

def test_add_floats():
    # assert add(0.1, 0.2) == 0.3 # 這裡可能會遇到浮點數精度問題，後面我們會討論
    assert abs(add(0.1, 0.2) - 0.3) < 1e-9
