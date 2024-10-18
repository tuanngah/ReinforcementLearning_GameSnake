from SnakeGame import SnakeGame
from Agent_Sarsa import SARSAAgent
import json
import matplotlib.pyplot as plt

# Tạo môi trường và agent SARSA
game = SnakeGame(render=False)
agent = SARSAAgent()

# Số lượng episodes cho việc huấn luyện
episodes = 5000
# Khởi tạo danh sách để lưu kết quả của mỗi episode
rewards = []
epsilons = []

# Huấn luyện agent
for episode in range(episodes):
    state = game.reset()
    total_reward = 0
    game_over = False

    # Lấy hành động đầu tiên
    action = agent.get_action(state)

    while not game_over:
        reward, game_over, score, next_state = game.play_step(action)
        next_action = agent.get_action(next_state)

        # Cập nhật giá trị Q bằng SARSA
        agent.update_q_value(state, action, reward, next_state, next_action)

        # Cập nhật state và action
        state = next_state
        action = next_action
        total_reward += reward

    agent.decay_epsilon()
    rewards.append(total_reward)
    epsilons.append(agent.epsilon)

    print(f"Episode: {episode+1}, Total Reward: {total_reward}, Score: {score}, Epsilon: {agent.epsilon}")

# Lưu Q-table sau khi huấn luyện
q_table_serializable = {str(state): actions for state, actions in agent.q_table.items()}
with open('q_table_sarsa.json', 'w') as json_file:
    json.dump(q_table_serializable, json_file, indent=4)

print("Training completed and Q-table saved!")

# Vẽ biểu đồ kết quả huấn luyện
def plot_training_results(rewards, epsilons):
    plt.figure(figsize=(12, 5))

    # Biểu đồ tổng phần thưởng của mỗi episode
    plt.subplot(1, 2, 1)
    plt.plot(rewards, color='b')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')

    # Biểu đồ giá trị epsilon của mỗi episode
    plt.subplot(1, 2, 2)
    plt.plot(epsilons, color='r')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay over Episodes')

    plt.tight_layout()
    plt.show()

# Vẽ biểu đồ quá trình huấn luyện
plot_training_results(rewards, epsilons)
