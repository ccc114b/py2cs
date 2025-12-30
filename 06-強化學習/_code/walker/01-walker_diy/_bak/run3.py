import gymnasium as gym
import torch
import numpy as np
import time
from model import ActorCritic

def run():
    # 使用 render_mode="human"
    env = gym.make("Walker2d-v5", render_mode="human")
    
    # 載入模型
    checkpoint = torch.load("ppo_walker_diy.pth", weights_only=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(state_dim, action_dim)
    model.load_state_dict(checkpoint['model'])
    rms_mean = checkpoint['rms_mean']
    rms_var = checkpoint['rms_var']
    model.eval()

    state, _ = env.reset()
    
    print("攝影機強行鎖定模式啟動...")

    try:
        while True:
            # 狀態歸一化
            norm_s = (state - rms_mean) / np.sqrt(rms_var + 1e-8)
            s_ts = torch.FloatTensor(norm_s).unsqueeze(0)
            
            with torch.no_grad():
                action = model.actor(s_ts)
            
            state, reward, term, trunc, _ = env.step(action.cpu().numpy().flatten())

            # -------------------------------------------------------------
            # 【核心修正】強行更新攝影機位置
            # -------------------------------------------------------------
            try:
                # 獲取軀幹 (torso) 的當前世界座標 (x, y, z)
                # Walker2d 主要在 XZ 平面運動，我們追蹤 X
                torso_x = env.unwrapped.data.body("torso").xpos[0]
                
                # 取得渲染器中的攝影機物件
                viewer = env.unwrapped.mujoco_renderer.default_cam
                
                # 將攝影機的焦點 (lookat) 設定在機器人的 X 座標上
                viewer.lookat[0] = torso_x
                viewer.distance = 5.0  # 拉開一點距離，防止跳太高時出鏡
                viewer.elevation = -20 # 調整俯視角度
            except Exception as e:
                pass # 防止某些版本環境屬性名稱不同導致報錯

            env.render()
            time.sleep(0.01)
            
            if term or trunc:
                state, _ = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    run()