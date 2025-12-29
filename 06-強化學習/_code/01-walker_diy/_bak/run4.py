import gymnasium as gym
import torch
import numpy as np
import time
from model import ActorCritic

def run():
    # 1. 建立環境
    env = gym.make("Walker2d-v5", render_mode="human")
    
    # 2. 載入模型
    checkpoint = torch.load("ppo_walker_diy.pth", weights_only=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(state_dim, action_dim)
    model.load_state_dict(checkpoint['model'])
    rms_mean = checkpoint['rms_mean']
    rms_var = checkpoint['rms_var']
    model.eval()

    state, _ = env.reset()
    
    print("相機鎖定模式已啟動。如果畫面沒跟隨，請嘗試在視窗內按 'Tab' 鍵切換視角。")

    try:
        while True:
            # 狀態歸一化
            norm_s = (state - rms_mean) / np.sqrt(rms_var + 1e-8)
            s_ts = torch.FloatTensor(norm_s).unsqueeze(0)
            
            with torch.no_grad():
                # 使用 actor 輸出的 tanh 均值
                action = model.actor(s_ts)
            
            # 執行動作
            action_np = action.cpu().numpy().flatten()
            state, reward, term, trunc, _ = env.step(action_np)

            # -------------------------------------------------------------
            # 【終極相機鎖定：底層強制同步】
            # -------------------------------------------------------------
            try:
                # 在 MuJoCo 中，Camera ID 0 通常是專門的追蹤攝影機
                # 我們強行讓渲染器每一幀都對齊第 1 號本體 (torso)
                viewer = env.unwrapped.mujoco_renderer
                
                # 設定追蹤模式：mjCAMERA_TRACKING (1)
                viewer.default_cam.type = 1 
                # 追蹤 body ID 1 (Walker2d 的軀幹)
                viewer.default_cam.trackbodyid = 1 
                
                # 設定一個固定的距離，防止機器人跳太高時出鏡
                viewer.default_cam.distance = 5.0 
            except:
                pass

            env.render()
            
            # 如果還是太快，增加延遲
            time.sleep(0.01)
            
            if term or trunc:
                print("Episode 結束，重置環境...")
                state, _ = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    run()