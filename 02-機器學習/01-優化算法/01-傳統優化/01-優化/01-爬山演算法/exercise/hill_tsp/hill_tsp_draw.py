import matplotlib.pyplot as plt
import random
import math

# 沿用之前的 TSPSolver 類別
class TSPSolver:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)

    def calculate_distance(self, path):
        total_dist = 0
        for i in range(self.num_cities):
            c1 = self.cities[path[i]]
            c2 = self.cities[path[(i + 1) % self.num_cities]]
            total_dist += math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        return total_dist

    def hill_climbing(self):
        current_path = list(range(self.num_cities))
        random.shuffle(current_path)
        current_dist = self.calculate_distance(current_path)

        while True:
            best_neighbor = None
            best_neighbor_dist = current_dist
            
            # 鄰域搜尋：嘗試所有可能的兩點交換
            for i in range(self.num_cities):
                for j in range(i + 1, self.num_cities):
                    neighbor = current_path[:]
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    dist = self.calculate_distance(neighbor)
                    if dist < best_neighbor_dist:
                        best_neighbor_dist = dist
                        best_neighbor = neighbor

            if best_neighbor is None:
                break
            current_path = best_neighbor
            current_dist = best_neighbor_dist
            
        return current_path, current_dist

def plot_tsp(cities, path, distance):
    x = [cities[i][0] for i in path] + [cities[path[0]][0]]
    y = [cities[i][1] for i in path] + [cities[path[0]][1]]
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o-', mfc='red', markersize=8, linewidth=2, label=f'Distance: {distance:.2f}')
    
    # 標記城市編號
    for i, (px, py) in enumerate(cities):
        plt.text(px, py + 0.2, f'City {i}', fontsize=12, ha='center')
        
    plt.title("TSP Solution using Hill Climbing")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 執行與測試 ---
if __name__ == "__main__":
    # 隨機產生 12 個城市
    random.seed(42) # 固定種子碼方便重複實驗
    random_cities = [(random.randint(0, 50), random.randint(0, 50)) for _ in range(12)]
    
    solver = TSPSolver(random_cities)
    
    # 執行多次隨機重啟以跳出局部最優
    best_path = None
    min_dist = float('inf')
    
    print("優化中...")
    for _ in range(20): 
        path, dist = solver.hill_climbing()
        if dist < min_dist:
            min_dist = dist
            best_path = path

    print(f"最優路徑: {best_path}")
    print(f"最優距離: {min_dist:.2f}")
    
    # 畫圖
    plot_tsp(random_cities, best_path, min_dist)