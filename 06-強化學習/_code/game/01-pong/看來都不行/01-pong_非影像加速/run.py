import gymnasium as gym
import ale_py
import numpy as np
import torch
from model import SimpleDQN, get_device, extract_pong_state
import time

# 註冊 ALE 環境
gym.register_envs(ale_py)

class PongRunner:
    def __init__(self, model_path="pong_best.pth", render=True):
        self.device = get_device()
        
        # 創建環境
        render_mode = "human" if render else None
        self.env = gym.make("ALE/Pong-v5", render_mode=render_mode)
        
        # 模型參數
        self.actions = [0, 2, 3]  # 0=NOOP, 2=UP, 3=DOWN
        self.n_actions = len(self.actions)
        self.state_size = 8
        
        # 載入模型
        self.model = SimpleDQN(self.state_size, self.n_actions).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found!")
            print("Please train the model first using train.py")
            exit(1)
        
        self.prev_obs = None
    
    def select_action(self, state):
        """選擇最佳動作（不使用隨機探索）"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_t)
            return q_values.argmax().item()
    
    def run(self, n_episodes=5, delay=0.01):
        """運行訓練好的模型"""
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            self.prev_obs = obs
            state = extract_pong_state(obs, None)
            
            episode_reward = 0
            done = False
            step_count = 0
            
            print(f"\n=== Episode {episode + 1} ===")
            
            while not done:
                # 選擇動作
                action_idx = self.select_action(state)
                action = self.actions[action_idx]
                
                # 執行動作
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 提取下一個狀態
                next_state = extract_pong_state(next_obs, self.prev_obs)
                
                episode_reward += reward
                step_count += 1
                
                self.prev_obs = obs
                obs = next_obs
                state = next_state
                
                # 控制幀率
                if delay > 0:
                    time.sleep(delay)
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1} finished")
            print(f"Steps: {step_count}")
            print(f"Total Reward: {episode_reward}")
        
        self.env.close()
        
        # 顯示統計資訊
        print("\n=== Summary ===")
        print(f"Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"Max Reward: {max(episode_rewards)}")
        print(f"Min Reward: {min(episode_rewards)}")
        
        return episode_rewards

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run trained Pong DQN agent')
    parser.add_argument('--model', type=str, default='pong_best.pth',
                        help='Path to trained model (default: pong_best.pth)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--delay', type=float, default=0.01,
                        help='Delay between frames in seconds (default: 0.01)')
    
    args = parser.parse_args()
    
    runner = PongRunner(model_path=args.model, render=not args.no_render)
    runner.run(n_episodes=args.episodes, delay=args.delay)

if __name__ == "__main__":
    main()