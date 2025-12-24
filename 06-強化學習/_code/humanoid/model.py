# model.py
from stable_baselines3 import PPO

def get_model(env, learning_rate=3e-4, verbose=1):
    """
    配置 PPO 模型。
    Humanoid 動作空間複雜（17個關節），需要較大的網路層。
    """
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]) # 策略網路與價值網路各兩層 256
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        device="auto" # 自動偵測 CUDA
    )
    return model
