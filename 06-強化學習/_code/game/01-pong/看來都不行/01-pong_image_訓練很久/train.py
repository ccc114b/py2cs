import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQN, ReplayBuffer, get_device
import cv2
from collections import deque

# 註冊 ALE 環境
gym.register_envs(ale_py)

class PongTrainer:
    def __init__(self, env_name="ALE/Pong-v5"):
        self.device = get_device()
        
        # 創建環境
        self.env = gym.make(env_name, render_mode=None)
        
        # 超參數
        self.n_actions = self.env.action_space.n
        self.input_shape = (4, 84, 84)  # 堆疊 4 幀灰階圖像
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 1000000
        self.target_update = 10000
        self.learning_rate = 0.0001
        self.replay_capacity = 100000
        self.min_replay_size = 10000
        
        # 創建網絡
        self.policy_net = DQN(self.input_shape, self.n_actions).to(self.device)
        self.target_net = DQN(self.input_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.replay_capacity)
        
        self.steps = 0
        
    def preprocess_frame(self, frame):
        """預處理遊戲幀"""
        # 轉換為灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 調整大小到 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # 正規化到 [0, 1]
        return resized.astype(np.float32) / 255.0
    
    def get_epsilon(self):
        """計算當前的 epsilon 值"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps / self.epsilon_decay)
    
    def select_action(self, state):
        """使用 epsilon-greedy 策略選擇動作"""
        epsilon = self.get_epsilon()
        
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()
    
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
        
        # 計算目標 Q 值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 計算損失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 優化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, n_episodes=10000, save_interval=100):
        """訓練 DQN"""
        episode_rewards = []
        best_reward = -float('inf')
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_frame(state)
            state_stack = deque([state] * 4, maxlen=4)
            
            episode_reward = 0
            done = False
            
            while not done:
                # 選擇動作
                current_state = np.array(state_stack)
                action = self.select_action(current_state)
                
                # 執行動作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 預處理下一幀
                next_state = self.preprocess_frame(next_state)
                state_stack.append(next_state)
                next_state_array = np.array(state_stack)
                
                # 存儲經驗
                self.replay_buffer.push(current_state, action, reward, next_state_array, done)
                
                episode_reward += reward
                self.steps += 1
                
                # 訓練模型
                if len(self.replay_buffer) >= self.min_replay_size:
                    loss = self.optimize_model()
                
                # 更新目標網絡
                if self.steps % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            
            # 打印進度
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.1f}, "
                      f"Avg(100): {avg_reward:.1f}, Epsilon: {self.get_epsilon():.3f}, "
                      f"Steps: {self.steps}")
            
            # 保存模型
            if episode % save_interval == 0 or avg_reward > best_reward:
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(self.policy_net.state_dict(), 'pong_best.pth')
                torch.save(self.policy_net.state_dict(), f'pong_saved.pth')
                print(f"Model saved at episode {episode}")
        
        self.env.close()

if __name__ == "__main__":
    trainer = PongTrainer()
    trainer.train()