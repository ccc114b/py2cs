# run.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time

def run():
    env_id = "Walker2d-v5"
    
    # 1. 重新建立環境，設定 render_mode 為 human
    env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])

    # 2. 載入訓練時的正規化參數
    try:
        env = VecNormalize.load("walker2d_vec_normalize.pkl", env)
        # 測試模式下，不要更新正規化統計數據，也不要正規化獎勵
        env.training = False
        env.norm_reward = False 
    except FileNotFoundError:
        print("錯誤：找不到 walker2d_vec_normalize.pkl，請先執行 train.py")
        return

    # 3. 載入 PPO 模型
    try:
        model = PPO.load("ppo_walker2d_model", env=env)
        print("模型載入成功，開始測試...")
    except FileNotFoundError:
        print("錯誤：找不到 ppo_walker2d_model.zip")
        return

    # 4. 執行測試
    obs = env.reset()
    for i in range(2000):
        # deterministic=True 讓動作更穩定
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        env.render()
        # 控制渲染速度，避免跑太快
        time.sleep(0.01)
        
        if dones:
            print(f"Episode 結束")
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    run()