# train.py
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from model import get_ppo_model

def train():
    # 建立環境 (建議使用 v5)
    env_id = "Walker2d-v5"
    
    # 封裝環境：SB3 要求使用 VecEnv 才能進行正規化
    env = DummyVecEnv([lambda: gym.make(env_id)])
    
    # 關鍵：加入 VecNormalize 正規化觀測值與獎勵
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    print(f"開始訓練 {env_id}...")
    model = get_ppo_model(env)
    
    # 訓練步數：1,000,000 步是 Walker2d 學會走路的最低標
    model.learn(total_timesteps=1000000, progress_bar=True)

    # 儲存模型
    model.save("ppo_walker2d_model")
    # 儲存正規化參數 (這在 run.py 必須用到，否則 AI 會摔倒)
    env.save("walker2d_vec_normalize.pkl")
    
    print("訓練完成！模型與正規化參數已儲存。")
    env.close()

if __name__ == "__main__":
    train()