# model.py
from stable_baselines3 import PPO

def get_ppo_model(env):
    """
    配置適合 MuJoCo Walker2d 的 PPO 模型
    """
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # 策略與價值網路
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,           # 每次更新前收集的步數
        batch_size=64,          # 小批量大小
        n_epochs=10,            # 每次數據重複訓練次數
        gamma=0.99,             # 折扣因子
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,           # 熵係數（若太快陷入局部最優，可設為 0.01）
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto"
    )
    return model