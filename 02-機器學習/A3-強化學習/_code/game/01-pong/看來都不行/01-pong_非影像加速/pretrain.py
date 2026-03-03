import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleDQN, get_device, extract_pong_state_from_ram

# 註冊 ALE 環境
gym.register_envs(ale_py)


class RuleBasedAgent:
    """基於規則的 Pong 代理：簡單追球策略"""
    
    def __init__(self):
        self.actions = [0, 2, 3]  # NOOP, UP, DOWN
    
    def select_action(self, state):
        """
        簡單規則：總是追球
        返回動作索引 (0, 1, 2)
        """
        ball_x, ball_y, ball_vx, ball_vy, paddle_y, enemy_paddle_y = state
        
        # 簡單追球邏輯
        if ball_y < paddle_y - 0.02:
            return 1  # UP
        elif ball_y > paddle_y + 0.02:
            return 2  # DOWN
        else:
            return 0  # STAY


def collect_expert_data(n_episodes=100, debug=False):
    """收集規則代理的經驗數據"""
    env = gym.make("ALE/Pong-v5", render_mode=None)
    agent = RuleBasedAgent()
    
    expert_data = []
    episode_rewards = []
    
    print("📚 收集專家數據（規則代理）...")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        prev_ram = None
        state = extract_pong_state_from_ram(obs, prev_ram)
        
        episode_reward = 0
        done = False
        step_count = 0
        action_counts = [0, 0, 0]
        
        while not done:
            # 規則選擇動作
            action_idx = agent.select_action(state)
            action = agent.actions[action_idx]
            action_counts[action_idx] += 1
            
            # Debug 模式
            if debug and episode == 0 and step_count < 10:
                print(f"\n  Step {step_count}:")
                print(f"    State: ball=({state[0]:.2f},{state[1]:.2f}), "
                      f"v=({state[2]:.3f},{state[3]:.3f}), "
                      f"paddle={state[4]:.2f}, enemy={state[5]:.2f}")
                print(f"    Action: {['STAY', 'UP', 'DOWN'][action_idx]}")
            
            # 執行動作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = extract_pong_state_from_ram(next_obs, obs)
            
            # 保存經驗
            expert_data.append((state, action_idx))
            
            episode_reward += reward
            prev_ram = obs
            obs = next_obs
            state = next_state
            step_count += 1
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  Episode {episode + 1}/{n_episodes}, "
                  f"Avg Reward: {avg_reward:.1f}, "
                  f"Actions: STAY={action_counts[0]}, UP={action_counts[1]}, DOWN={action_counts[2]}")
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    print(f"\n✓ 收集完成！共 {len(expert_data)} 個樣本")
    print(f"  規則代理平均獎勵: {avg_reward:.2f}")
    
    return expert_data, avg_reward


def pretrain_from_expert(expert_data, n_epochs=10):
    """從專家數據預訓練模型"""
    device = get_device()
    
    # 創建模型
    model = SimpleDQN(state_size=6, n_actions=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 準備數據
    states = np.array([s for s, a in expert_data])
    actions = np.array([a for s, a in expert_data])
    
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    
    print(f"\n🎓 開始預訓練（模仿學習）...")
    print(f"  訓練樣本: {len(expert_data)}")
    print(f"  Epochs: {n_epochs}")
    
    batch_size = 256
    n_batches = len(expert_data) // batch_size
    
    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        
        # 隨機打亂數據
        indices = torch.randperm(len(expert_data))
        
        for i in range(n_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            
            # 前向傳播
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 計算準確率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_actions).sum().item()
        
        avg_loss = total_loss / n_batches
        accuracy = 100 * correct / (n_batches * batch_size)
        
        print(f"  Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print(f"\n✓ 預訓練完成！")
    
    return model


def test_pretrained_model(model, n_episodes=10):
    """測試預訓練模型的表現"""
    device = get_device()
    env = gym.make("ALE/Pong-v5", render_mode=None)
    actions = [0, 2, 3]
    
    episode_rewards = []
    
    print(f"\n🎮 測試預訓練模型...")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        prev_ram = None
        state = extract_pong_state_from_ram(obs, prev_ram)
        
        episode_reward = 0
        done = False
        
        while not done:
            # 使用模型選擇動作
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_t)
                action_idx = q_values.argmax().item()
                action = actions[action_idx]
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = extract_pong_state_from_ram(next_obs, obs)
            
            episode_reward += reward
            prev_ram = obs
            obs = next_obs
            state = next_state
        
        episode_rewards.append(episode_reward)
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    print(f"  預訓練模型平均獎勵: {avg_reward:.2f}")
    
    return avg_reward


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pong 預訓練：使用規則代理初始化模型")
    print("=" * 60)
    
    # 步驟 1：收集專家數據
    expert_data, rule_reward = collect_expert_data(n_episodes=50, debug=args.debug)
    
    # 步驟 2：預訓練模型
    model = pretrain_from_expert(expert_data, n_epochs=20)
    
    # 步驟 3：測試預訓練模型
    pretrain_reward = test_pretrained_model(model, n_episodes=10)
    
    # 步驟 4：保存預訓練模型
    torch.save(model.state_dict(), 'pong_pretrained.pth')
    print(f"\n💾 模型已保存到 'pong_pretrained.pth'")
    
    # 總結
    print("\n" + "=" * 60)
    print("📊 總結")
    print("=" * 60)
    print(f"  規則代理獎勵:     {rule_reward:.2f}")
    print(f"  預訓練模型獎勵:   {pretrain_reward:.2f}")
    
    if rule_reward > -20:
        print(f"\n✅ 預訓練成功！規則代理表現良好")
        print(f"💡 現在可以運行: python train.py --pretrain")
    else:
        print(f"\n⚠️  規則代理表現不佳（{rule_reward:.2f}）")
        print(f"💡 建議直接運行: python train.py")
    
    print("=" * 60)


if __name__ == "__main__":
    main()