# 修改後的 run.py 核心邏輯
import gymnasium as gym
import torch
import numpy as np
import time
from model import ActorCritic

def run():
    env = gym.make("Walker2d-v5", render_mode="human")
    
    # 載入模型與權重 (加上新版所需的權限)
    checkpoint = torch.load("ppo_walker_diy.pth", weights_only=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(state_dim, action_dim)
    model.load_state_dict(checkpoint['model'])
    rms_mean = checkpoint['rms_mean']
    rms_var = checkpoint['rms_var']
    model.eval()

    state, _ = env.reset()
    
    # -------------------------------------------------------------
    # 獲取 MuJoCo 內部模型數據，以便我們手動控制攝影機
    # -------------------------------------------------------------
    model_data = env.unwrapped.model
    data = env.unwrapped.data
    
    # 找到 torso 的 body ID (通常是 1，但這樣找最準)
    torso_id = 0
    for i in range(model_data.nbody):
        if model_data.body(i).name == "torso":
            torso_id = i
            break

    print(f"攝影機已鎖定 Torso (ID: {torso_id})")

    while True:
        # 狀態歸一化
        norm_s = (state - rms_mean) / np.sqrt(rms_var + 1e-8)
        s_ts = torch.FloatTensor(norm_s).unsqueeze(0)
        
        with torch.no_grad():
            action = model.actor(s_ts)
            
        state, reward, term, trunc, _ = env.step(action.cpu().numpy().flatten())
        
        # ---------------------------------------------------------
        # 強制更新攝影機：讓它每一幀都看著 torso 的位置
        # ---------------------------------------------------------
        try:
            # 在 human 模式下，我們可以直接修改 viewer 的攝影機視點
            # env.unwrapped.mujoco_renderer 是 Gymnasium 負責渲染的物件
            viewer = env.unwrapped.mujoco_renderer
            
            # 設定攝影機為追蹤模式 (Mode 1: mjCAMERA_TRACKING)
            # 並且指向 torso 的 body ID
            viewer.default_cam.type = 1 
            viewer.default_cam.trackbodyid = torso_id
            # 調整距離，不要離太近
            viewer.default_cam.distance = 4.0 
        except Exception:
            pass
            
        env.render()
        time.sleep(0.02) # 稍微慢一點，比較好觀察
        
        if term or trunc:
            state, _ = env.reset()

if __name__ == "__main__":
    run()