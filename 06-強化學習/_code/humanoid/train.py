# train.py
import gymnasium as gym
from model import get_model
import os

def train():
    # 建立環境
    env = gym.make("Humanoid-v4")
    
    # 取得模型
    model = get_model(env)
    
    # 設定訓練步數 (建議至少 2,000,000)
    total_timesteps = 5000000 #2000000
    
    print("開始訓練...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # 儲存模型
    model_path = "humanoid_ppo_model"
    model.save(model_path)
    print(f"訓練完成，模型已儲存至 {model_path}")
    
    env.close()

if __name__ == "__main__":
    train()
