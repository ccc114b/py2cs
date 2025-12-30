# run.py
import gymnasium as gym
from stable_baselines3 import PPO
import time
import sys

def run(env_name):
    # 建立環境，設定 render_mode 為 human 才能看到畫面
    env = gym.make(env_name, render_mode="human")
    
    # 載入模型
    model_path = env_name + ".zip"
    try:
        model = PPO.load(model_path, env=env)
        print("成功載入模型！")
    except:
        print("找不到模型檔案，請先執行 train.py")
        return

    # 執行測試
    obs, info = env.reset()
    for _ in range(2000):
        # 使用模型預測動作
        action, _states = model.predict(obs, deterministic=True)
        
        # 執行動作
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
            print("Episode 結束，重置環境...")
            time.sleep(0.5)

    env.close()

if __name__ == "__main__":
    run(sys.argv[1])
