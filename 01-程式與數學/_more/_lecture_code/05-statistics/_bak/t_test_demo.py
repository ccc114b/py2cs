import numpy as np
from scipy import stats

def perform_t_test(group_a, group_b, equal_var=False):
    """
    對兩組獨立樣本執行獨立樣本 t 檢定。

    Args:
        group_a (list or numpy.array): 第一組樣本數據。
        group_b (list or numpy.array): 第二組樣本數據。
        equal_var (bool): 是否假設兩組的變異數相等。
                          這裡我們設定為 False，執行 Welch's t-test，
                          通常更為穩健，且不需要假設變異數相等。

    Returns:
        scipy.stats.Ttest_indResult: t 檢定的結果物件，包含 t 統計量和 p 值。
    """
    # 將輸入轉換為 numpy 陣列以確保兼容性
    group_a = np.array(group_a)
    group_b = np.array(group_b)

    # 執行獨立樣本 t 檢定
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=equal_var)
    
    # 返回結果物件
    return stats.Ttest_indResult(statistic=t_stat, pvalue=p_value)

if __name__ == "__main__":
    # 範例數據：A 組的療效分數顯著高於 B 組
    group_a_scores = [85, 90, 88, 92, 86, 95]
    group_b_scores = [70, 75, 72, 78, 73, 76]

    # 執行 t 檢定
    result = perform_t_test(group_a_scores, group_b_scores)
    
    print("--- 統計分析結果 ---")
    print(f"A 組平均分數: {np.mean(group_a_scores):.2f}")
    print(f"B 組平均分數: {np.mean(group_b_scores):.2f}")
    print(f"t 統計量: {result.statistic:.4f}")
    print(f"p 值: {result.pvalue:.4f}")

    # 決策判斷
    alpha = 0.05
    if result.pvalue < alpha:
        print(f"\np 值 ({result.pvalue:.4f}) 小於顯著水準 ({alpha})。")
        print("我們有足夠的證據拒絕虛無假設，認為兩組的平均值存在顯著差異。")
    else:
        print(f"\np 值 ({result.pvalue:.4f}) 大於顯著水準 ({alpha})。")
        print("我們沒有足夠的證據拒絕虛無假設，無法證明兩組的平均值存在顯著差異。")