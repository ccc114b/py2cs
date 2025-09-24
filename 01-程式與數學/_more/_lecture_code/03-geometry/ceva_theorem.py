# ceva.py

import math
from geometry_objects import Point, Line, EPSILON

class CevaPointSet:
    """
    表示一個用於塞瓦定理驗證的點集：
    - 三個三角形頂點 A, B, C
    - 對邊上的三個點 D, E, F
    """
    def __init__(self, A, B, C, D, E, F):
        if not all(isinstance(p, Point) for p in [A, B, C, D, E, F]):
            raise TypeError("All points must be Point objects.")

        self.A = A
        self.B = B
        self.C = C
        self.D = D  # D is on BC
        self.E = E  # E is on CA
        self.F = F  # F is on AB

    def get_ratios(self):
        """
        計算塞瓦定理中的三個比值：AF/FB, BD/DC, CE/EA
        
        回傳:
            tuple: (AF/FB, BD/DC, CE/EA)
        """
        # 注意：需要處理分母為零（即點在端點上）的情況
        af = self.A.distance_to(self.F)
        fb = self.F.distance_to(self.B)
        ratio_af_fb = af / fb if fb > EPSILON else float('inf')

        bd = self.B.distance_to(self.D)
        dc = self.D.distance_to(self.C)
        ratio_bd_dc = bd / dc if dc > EPSILON else float('inf')

        ce = self.C.distance_to(self.E)
        ea = self.E.distance_to(self.A)
        ratio_ce_ea = ce / ea if ea > EPSILON else float('inf')
        
        return ratio_af_fb, ratio_bd_dc, ratio_ce_ea

def is_concurrent(ceva_points):
    """
    根據塞瓦定理，判斷三條塞瓦線 AD, BE, CF 是否共點。

    Args:
        ceva_points (CevaPointSet): 包含三角形頂點和對邊分點的物件。

    回傳:
        bool: 如果三條線共點，回傳 True；否則回傳 False。
    """
    if not isinstance(ceva_points, CevaPointSet):
        raise TypeError("Input must be a CevaPointSet object.")

    try:
        ratio_af_fb, ratio_bd_dc, ratio_ce_ea = ceva_points.get_ratios()
        product = ratio_af_fb * ratio_bd_dc * ratio_ce_ea
        
        return math.isclose(product, 1.0, abs_tol=EPSILON)
    
    except ZeroDivisionError:
        # 處理點在端點上的極端情況，此時比值為無窮大，積可能不為1
        return False
