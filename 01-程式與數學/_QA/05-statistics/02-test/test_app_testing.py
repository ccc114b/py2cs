import pytest
import numpy as np
from scipy import stats
from app_testing import conduct_t_test

def test_t_test_significant_difference():
    """
    測試當新 App 成績顯著高於傳統組時，p 值是否小於 0.05。
    """
    new_app_scores = [85, 90, 88, 92, 86, 95]
    traditional_scores = [70, 75, 72, 78, 73, 76]
    
    t_stat, p_value = conduct_t_test(new_app_scores, traditional_scores)
    
    # 斷言 p 值小於 0.05 (顯著水準)
    assert p_value < 0.05
    # 斷言 t 統計量為正值，因為新 App 成績高於傳統組
    assert t_stat > 0

def test_t_test_no_significant_difference():
    """
    測試當新 App 成績沒有顯著差異時，p 值是否大於 0.05。
    """
    new_app_scores = [75, 78, 80, 76, 79, 81]
    traditional_scores = [76, 79, 74, 80, 77, 75]

    t_stat, p_value = conduct_t_test(new_app_scores, traditional_scores)
    
    # 斷言 p 值大於 0.05
    assert p_value > 0.05

def test_conduct_t_test_with_insufficient_data():
    """
    測試當輸入的樣本數不足時，是否會拋出 ValueError。
    """
    # 只有一個樣本點
    with pytest.raises(ValueError):
        conduct_t_test([10], [20, 30])
    
    # 兩組都只有一個樣本點
    with pytest.raises(ValueError):
        conduct_t_test([10], [20])

def test_t_test_with_known_values():
    """
    使用已知的 Scipy t 檢定結果來驗證我們的函數。
    """
    group_a = np.array([1, 2, 3, 4, 5])
    group_b = np.array([6, 7, 8, 9, 10])

    # 我們自己的函數
    t_stat, p_value = conduct_t_test(group_a, group_b)

    # Scipy 內建的 t 檢定，作為黃金標準
    scipy_result = stats.ttest_ind(group_a, group_b, equal_var=False, alternative='greater')
    
    # 斷言我們的函數結果與 Scipy 的結果在小數點後 5 位相同
    np.testing.assert_almost_equal(t_stat, scipy_result.statistic, decimal=5)
    np.testing.assert_almost_equal(p_value, scipy_result.pvalue, decimal=5)