import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_device():
    """
    自動選擇最佳的計算設備
    優先順序: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


class SimpleDQN(nn.Module):
    """簡化的 DQN，使用提取的狀態特徵而非原始像素"""
    def __init__(self, state_size=6, n_actions=3, hidden_size=256):
        super(SimpleDQN, self).__init__()
        
        # 更深的網絡以學習複雜策略
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def extract_pong_state_from_ram(obs, prev_obs=None):
    """
    從 Pong 觀察提取狀態（簡化版）
    obs 是 210x160x3 的 RGB 圖像
    
    返回: [ball_x, ball_y, ball_vx, ball_vy, paddle_y, enemy_paddle_y]
    """
    import cv2
    
    # 轉灰階
    if len(obs.shape) == 3:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    else:
        gray = obs
    
    # 裁剪遊戲區域（去掉上下邊框和分數）
    game = gray[34:194, 16:144]  # Pong 遊戲區域
    h, w = game.shape
    
    # 二值化
    _, binary = cv2.threshold(game, 100, 255, cv2.THRESH_BINARY)
    
    # 找球：在中間區域尋找小的白色物體
    ball_x, ball_y = 0.5, 0.5
    
    # 簡單方法：掃描每一列，找到最小的連續白色區域
    for x in range(20, w-20):
        col = binary[:, x]
        white_regions = []
        start = -1
        for y in range(len(col)):
            if col[y] > 0 and start == -1:
                start = y
            elif col[y] == 0 and start != -1:
                white_regions.append((start, y))
                start = -1
        
        # 找最小的區域（可能是球）
        if white_regions:
            smallest = min(white_regions, key=lambda r: r[1] - r[0])
            if smallest[1] - smallest[0] <= 6:  # 球很小
                ball_x = x / w
                ball_y = (smallest[0] + smallest[1]) / 2 / h
                break
    
    # 找球拍：左右兩側的長條
    # 左側（玩家）
    left = binary[:, :5]
    left_y = np.where(left > 0)
    if len(left_y[0]) > 10:
        paddle_y = np.mean(left_y[0]) / h
    else:
        paddle_y = 0.5
    
    # 右側（敵人）
    right = binary[:, -5:]
    right_y = np.where(right > 0)
    if len(right_y[0]) > 10:
        enemy_paddle_y = np.mean(right_y[0]) / h
    else:
        enemy_paddle_y = 0.5
    
    # 計算速度
    ball_vx, ball_vy = 0.0, 0.0
    if prev_obs is not None:
        # 重複相同的提取邏輯而不是遞迴調用
        prev_gray = cv2.cvtColor(prev_obs, cv2.COLOR_RGB2GRAY) if len(prev_obs.shape) == 3 else prev_obs
        prev_game = prev_gray[34:194, 16:144]
        _, prev_binary = cv2.threshold(prev_game, 100, 255, cv2.THRESH_BINARY)
        
        prev_ball_x, prev_ball_y = 0.5, 0.5
        for x in range(20, w-20):
            col = prev_binary[:, x]
            white_regions = []
            start = -1
            for y in range(len(col)):
                if col[y] > 0 and start == -1:
                    start = y
                elif col[y] == 0 and start != -1:
                    white_regions.append((start, y))
                    start = -1
            
            if white_regions:
                smallest = min(white_regions, key=lambda r: r[1] - r[0])
                if smallest[1] - smallest[0] <= 6:
                    prev_ball_x = x / w
                    prev_ball_y = (smallest[0] + smallest[1]) / 2 / h
                    break
        
        ball_vx = ball_x - prev_ball_x
        ball_vy = ball_y - prev_ball_y
        ball_vx = np.clip(ball_vx, -0.1, 0.1)
        ball_vy = np.clip(ball_vy, -0.1, 0.1)
    
    return np.array([ball_x, ball_y, ball_vx, ball_vy, paddle_y, enemy_paddle_y], dtype=np.float32)


class ReplayBuffer:
    """經驗回放緩衝區"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)