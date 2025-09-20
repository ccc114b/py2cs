import numpy as np
from scipy import stats

def conduct_t_test(new_app_scores, traditional_scores):
    """
    對兩組學生的數學成績進行獨立樣本 t 檢定。

    Args:
        new_app_scores (list or numpy.array): 使用新 App 的學生分數。
        traditional_scores (list or numpy.array): 使用傳統教學方法的學生分數。

    Returns:
        tuple: 包含 t 統計量和 p 值。
    """
    if len(new_app_scores) < 2 or len(traditional_scores) < 2:
        raise ValueError("兩組樣本數都必須大於 1")
    
    # 使用獨立樣本 t 檢定 (假設變異數不相等，使用 Welch's t-test 更穩健)
    t_stat, p_value = stats.ttest_ind(
        new_app_scores, 
        traditional_scores, 
        equal_var=False, 
        alternative='greater' # 這裡指定對立假設為 'greater' (單尾檢定)
    )
    
    return t_stat, p_value

def make_decision(p_value, alpha=0.05):
    """
    根據 p 值和顯著水準做出檢定決策。

    Args:
        p_value (float): t 檢定得到的 p 值。
        alpha (float): 顯著水準，通常為 0.05。

    Returns:
        str: 檢定結果的文字描述。
    """
    if p_value < alpha:
        return f"p 值 ({p_value:.4f}) 小於顯著水準 ({alpha})，有足夠證據拒絕虛無假設。結論：新 App 顯著提高了學生成績。"
    else:
        return f"p 值 ({p_value:.4f}) 大於顯著水準 ({alpha})，沒有足夠證據拒絕虛無假設。結論：新 App 沒有顯著效果。"

# 範例執行
if __name__ == "__main__":
    # 模擬數據：新 App 組的成績分數顯著較高
    new_app_scores = [85, 90, 88, 92, 86, 95]
    traditional_scores = [70, 75, 72, 78, 73, 76]
    
    t_stat, p_value = conduct_t_test(new_app_scores, traditional_scores)
    result_text = make_decision(p_value)

    print("--- 檢定結果 ---")
    print(f"t 統計量: {t_stat:.4f}")
    print(f"p 值: {p_value:.4f}")
    print(result_text)

    # 模擬另一組數據：新 App 組的成績分數沒有顯著差異
    new_app_scores_no_diff = [75, 78, 80, 76, 79, 81]
    traditional_scores_no_diff = [76, 79, 74, 80, 77, 75]
    
    t_stat_no_diff, p_value_no_diff = conduct_t_test(
        new_app_scores_no_diff, 
        traditional_scores_no_diff
    )
    result_text_no_diff = make_decision(p_value_no_diff)
    
    print("\n--- 另一組檢定結果 ---")
    print(f"t 統計量: {t_stat_no_diff:.4f}")
    print(f"p 值: {p_value_no_diff:.4f}")
    print(result_text_no_diff)