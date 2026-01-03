import numpy as np
from scipy import stats
from t_test_demo import perform_t_test

def test_t_test_significant_difference():
    """
    測試當兩組數據有顯著差異時，p 值是否夠小。
    """
    # 設置兩組數據，讓 A 組的平均數明顯高於 B 組
    group_a = [10, 11, 12, 13, 14]
    group_b = [5, 6, 7, 8, 9]
    
    result = perform_t_test(group_a, group_b)
    
    # 斷言 p 值小於 0.05 (顯著水準)
    assert result.pvalue < 0.05
    # 斷言 t 統計量為正值，因為 A 組平均數大於 B 組
    assert result.statistic > 0

def test_t_test_no_significant_difference():
    """
    測試當兩組數據沒有顯著差異時，p 值是否夠大。
    """
    # 設置兩組數據，它們的平均數非常接近
    group_a = [10, 11, 12, 13, 14]
    group_b = [11, 12, 11, 10, 13]

    result = perform_t_test(group_a, group_b)
    
    # 斷言 p 值大於 0.05
    assert result.pvalue > 0.05

def test_t_test_with_known_values():
    """
    使用 Scipy 文檔或已知結果來驗證我們的函數。
    這個測試可以確保我們的函數與 Scipy 內建函數的結果一致。
    """
    # 創建兩個隨機但具有相同平均數的樣本
    rng = np.random.default_rng(0)
    group_a = rng.normal(loc=10, scale=1, size=10)
    group_b = rng.normal(loc=10, scale=1, size=10)

    # 用我們的函數計算
    our_result = perform_t_test(group_a, group_b)

    # 用 Scipy 內建函數直接計算，作為黃金標準 (gold standard)
    scipy_result = stats.ttest_ind(group_a, group_b, equal_var=False)

    # 斷言我們的函數結果與 Scipy 的結果在小數點後 5 位相同
    np.testing.assert_almost_equal(our_result.statistic, scipy_result.statistic, decimal=5)
    np.testing.assert_almost_equal(our_result.pvalue, scipy_result.pvalue, decimal=5)