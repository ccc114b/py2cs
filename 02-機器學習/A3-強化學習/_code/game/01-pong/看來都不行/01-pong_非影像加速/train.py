import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleDQN, ReplayBuffer, get_device, extract_pong_state_from_ram

# 註冊 ALE 環境
gym.register_envs(ale_py)

class PongTrainer:
    def __init__(self, env_name="ALE/Pong-v5", debug=False, pretrain_path=None):
        self.device = get_device()
        self.debug = debug
        
        # 創建環境
        self.env = gym.make(env_name, render_mode=None)
        
        # 簡化的動作空間：只用 STAY, UP, DOWN
        self.actions = [0, 2, 3]  # 0=NOOP, 2=UP, 3=DOWN
        self.n_actions = len(self.actions)
        self.state_size = 6  # [ball_x, ball_y, ball_vx, ball_vy, paddle_y, enemy_paddle_y]
        
        # 超參數（優化過）
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1  # 提高最低探索率
        self.epsilon_decay = 20000  # 放慢衰減
        self.target_update = 1000  # 降低更新頻率
        self.learning_rate = 0.0003  # 降低學習率
        self.replay_capacity = 50000
        self.min_replay_size = 1000
        
        # 創建網絡
        self.policy_net = SimpleDQN(self.state_size, self.n_actions).to(self.device)
        self.target_net = SimpleDQN(self.state_size, self.n_actions).to(self.device)
        
        # 載入預訓練模型（如果有）
        if pretrain_path:
            try:
                self.policy_net.load_state_dict(torch.load(pretrain_path, map_location=self.device))
                print(f"✓ 載入預訓練模型: {pretrain_path}")
            except FileNotFoundError:
                print(f"⚠️  找不到預訓練模型 {pretrain_path}，將從零開始訓練")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.replay_capacity)
        
        self.steps = 0
        self.prev_ram = None
        
    def get_epsilon(self):
        """計算當前的 epsilon 值"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps / self.epsilon_decay)
    
    def select_action(self, state):
        """使用 epsilon-greedy 策略選擇動作"""
        epsilon = self.get_epsilon()
        
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()
    
    def shape_reward(self, reward, state, next_state, done):
        """
        獎勵塑形：給予中間獎勵以加速學習（調整後的版本）
        """
        shaped_reward = reward * 5  # 減少放大倍數
        
        if not done and reward == 0:  # 只在沒得分時給予塑形獎勵
            # 獎勵：球拍靠近球
            ball_y = state[1]
            paddle_y = state[4]
            next_ball_y = next_state[1]
            next_paddle_y = next_state[4]
            
            prev_dist = abs(ball_y - paddle_y)
            next_dist = abs(next_ball_y - next_paddle_y)
            
            # 如果球拍靠近球，給予小獎勵（減小幅度）
            if next_dist < prev_dist:
                shaped_reward += 0.05
            
            # 輕微懲罰距離太遠（減小懲罰）
            if next_dist > 0.3:
                shaped_reward -= 0.01
        
        return shaped_reward
    
    def optimize_model(self):
        """執行一步優化"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 轉換為張量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 計算當前 Q 值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: 使用 policy network 選擇動作，target network 評估
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 計算損失
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        # 優化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, n_episodes=1000, save_interval=50):
        """訓練 DQN"""
        episode_rewards = []
        episode_raw_rewards = []
        best_avg_reward = -float('inf')
        
        print("\n開始訓練改進版 Pong DQN...")
        print("✓ 改進的視覺狀態提取")
        print("✓ 獎勵塑形（reward shaping）")
        print("✓ Double DQN\n")
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            self.prev_ram = obs
            state = extract_pong_state_from_ram(obs, None)
            
            episode_reward = 0
            raw_reward = 0
            done = False
            losses = []
            hits = 0  # 記錄擊球次數
            
            while not done:
                # 選擇動作
                action_idx = self.select_action(state)
                action = self.actions[action_idx]
                
                # 執行動作
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 提取下一個狀態
                next_state = extract_pong_state_from_ram(next_obs, self.prev_ram)
                
                # 獎勵塑形
                shaped_reward = self.shape_reward(reward, state, next_state, done)
                
                # 記錄原始獎勵
                raw_reward += reward
                if reward > 0:
                    hits += 1
                
                # 存儲經驗
                self.replay_buffer.push(state, action_idx, shaped_reward, next_state, done)
                
                episode_reward += shaped_reward
                self.steps += 1
                
                # 訓練模型
                if len(self.replay_buffer) >= self.min_replay_size:
                    loss = self.optimize_model()
                    if loss is not None:
                        losses.append(loss)
                
                # 更新目標網絡
                if self.steps % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                self.prev_ram = obs
                obs = next_obs
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_raw_rewards.append(raw_reward)
            
            avg_reward = np.mean(episode_rewards[-100:])
            avg_raw_reward = np.mean(episode_raw_rewards[-100:])
            avg_loss = np.mean(losses) if losses else 0
            
            # 打印進度
            if episode % 10 == 0:
                trend = ""
                if len(episode_raw_rewards) >= 20:
                    recent_20 = np.mean(episode_raw_rewards[-20:])
                    prev_20 = np.mean(episode_raw_rewards[-40:-20]) if len(episode_raw_rewards) >= 40 else recent_20
                    if recent_20 > prev_20 + 0.5:
                        trend = "📈"
                    elif recent_20 < prev_20 - 0.5:
                        trend = "📉"
                    else:
                        trend = "➡️"
                
                print(f"Ep {episode:4d} | "
                      f"Raw: {raw_reward:3.0f} | "
                      f"Shaped: {episode_reward:7.1f} | "
                      f"Avg Raw(100): {avg_raw_reward:6.2f} {trend} | "
                      f"ε: {self.get_epsilon():.3f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Hits: {hits} | "
                      f"Buffer: {len(self.replay_buffer)}")
            
            # Debug 模式：顯示狀態信息
            if self.debug and episode % 50 == 0:
                print(f"\n  [Debug] Episode {episode} 最後一幀狀態:")
                print(f"    Ball: ({state[0]:.2f}, {state[1]:.2f}), V: ({state[2]:.3f}, {state[3]:.3f})")
                print(f"    Paddle: {state[4]:.2f}, Enemy: {state[5]:.2f}")
            
            # 保存模型（基於原始獎勵）
            if episode % save_interval == 0 or avg_raw_reward > best_avg_reward:
                if avg_raw_reward > best_avg_reward:
                    best_avg_reward = avg_raw_reward
                    torch.save(self.policy_net.state_dict(), 'pong_best.pth')
                    print(f"  ⭐ 新的最佳平均原始獎勵: {best_avg_reward:.2f}")
                torch.save(self.policy_net.state_dict(), f'pong_ep{episode}.pth')
        
        self.env.close()
        print(f"\n訓練完成！最佳平均原始獎勵: {best_avg_reward:.2f}")
        
        # 顯示訓練曲線建議
        print("\n💡 提示：你可以用以下指令測試模型：")
        print("   python run.py --model pong_best.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--pretrain', action='store_true', help='Load pretrained model')
    parser.add_argument('--pretrain-path', type=str, default='pong_pretrained.pth', 
                        help='Path to pretrained model')
    args = parser.parse_args()
    
    pretrain_path = args.pretrain_path if args.pretrain else None
    
    trainer = PongTrainer(debug=args.debug, pretrain_path=pretrain_path)
    trainer.train(n_episodes=args.episodes)