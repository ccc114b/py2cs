
```sh
(py310) cccimac@cccimacdeiMac 06-space % pytest 
============================================= test session starts ==============================================
platform darwin -- Python 3.10.16, pytest-8.3.4, pluggy-1.5.0
rootdir: /Users/cccimac/Desktop/ccc/py2cs/01-程式與數學/_homework/06-space
plugins: anyio-4.9.0
collected 13 items                                                                                             

test_normed_vector_space_axioms.py .............                                                         [100%]

============================================== 13 passed in 0.02s ==============================================
(py310) cccimac@cccimacdeiMac 06-space % python test_normed_vector_space_axioms.py 
============================================= test session starts ==============================================
platform darwin -- Python 3.10.16, pytest-8.3.4, pluggy-1.5.0 -- /opt/homebrew/Caskroom/miniforge/base/envs/py310/bin/python
cachedir: .pytest_cache
rootdir: /Users/cccimac/Desktop/ccc/py2cs/01-程式與數學/_homework/06-space
plugins: anyio-4.9.0
collected 13 items                                                                                             

test_normed_vector_space_axioms.py::test_vector_addition_closure PASSED                                  [  7%]
test_normed_vector_space_axioms.py::test_vector_addition_associativity PASSED                            [ 15%]
test_normed_vector_space_axioms.py::test_vector_additive_identity PASSED                                 [ 23%]
test_normed_vector_space_axioms.py::test_vector_additive_inverse PASSED                                  [ 30%]
test_normed_vector_space_axioms.py::test_scalar_multiplication_closure PASSED                            [ 38%]
test_normed_vector_space_axioms.py::test_scalar_multiplication_associativity PASSED                      [ 46%]
test_normed_vector_space_axioms.py::test_scalar_multiplication_identity PASSED                           [ 53%]
test_normed_vector_space_axioms.py::test_distributivity_scalar_over_vector_addition PASSED               [ 61%]
test_normed_vector_space_axioms.py::test_distributivity_vector_over_scalar_addition PASSED               [ 69%]
test_normed_vector_space_axioms.py::test_norm_non_negativity PASSED                                      [ 76%]
test_normed_vector_space_axioms.py::test_norm_positive_definiteness PASSED                               [ 84%]
test_normed_vector_space_axioms.py::test_norm_homogeneity PASSED                                         [ 92%]
test_normed_vector_space_axioms.py::test_norm_triangle_inequality PASSED                                 [100%]

============================================== 13 passed in 0.01s ==============================================

--- Demonstrating a Cauchy Sequence (Conceptual) ---
x_1 = Vector2D(0.0, 0.0)
x_2 = Vector2D(0.25, 0.25)
x_3 = Vector2D(0.375, 0.375)
x_4 = Vector2D(0.4375, 0.4375)
x_5 = Vector2D(0.46875, 0.46875)
Distance between x_1 and x_2: 0.353553
Distance between x_1 and x_3: 0.530330
Distance between x_1 and x_4: 0.618718
Distance between x_1 and x_5: 0.662913
Distance between x_2 and x_3: 0.176777
Distance between x_2 and x_4: 0.265165
Distance between x_2 and x_5: 0.309359
Distance between x_3 and x_4: 0.088388
Distance between x_3 and x_5: 0.132583
Distance between x_4 and x_5: 0.044194
```
