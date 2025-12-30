import gymnasium as gym
import torch
import numpy as np
import time
from model import ActorCritic

# 允許 PyTorch 載入 NumPy 的資料結構
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype])

def run():
    env = gym.make("Walker2d-v5", render_mode="human")
    # 攝影機追蹤
    env.unwrapped.camera_id = 0 
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(state_dim, action_dim)
    
    checkpoint = torch.load("ppo_walker_diy.pth", weights_only=False)
    model.load_state_dict(checkpoint['model'])
    rms_mean = checkpoint['rms_mean']
    rms_var = checkpoint['rms_var']
    model.eval()

    state, _ = env.reset()
    while True:
        # 使用訓練時的統計數據進行歸一化
        norm_s = (state - rms_mean) / np.sqrt(rms_var + 1e-8)
        s_ts = torch.FloatTensor(norm_s).unsqueeze(0)
        
        with torch.no_grad():
            action = model.actor(s_ts) # 使用均值動作
            
        state, reward, term, trunc, _ = env.step(action.cpu().numpy().flatten())
        env.render()
        time.sleep(0.01)
        
        if term or trunc:
            state, _ = env.reset()

if __name__ == "__main__":
    run()