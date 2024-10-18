from SnakeGame import SnakeGame
from Agent_MonteCarlo import MonteCarloAgent
import json
import matplotlib.pyplot as plt

# Tạo môi trường và agent Monte Carlo
game = SnakeGame(render=False)
agent = MonteCarloAgent()

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

    while not game_over:
        # Lựa chọn hành động
        action = agent.get_action(state)
        reward, game_over, score, next_state = game.play_step(action)

        # Lưu trữ episode
        agent.store_episode(state, action, reward)

        # Chuyển sang trạng thái tiếp theo
        state = next_state
        total_reward += reward

    # Cập nhật giá trị Q sau khi kết thúc một episode
    agent.update_q_values()

    # Giảm epsilon
    agent.decay_epsilon()

    rewards.append(total_reward)
    epsilons.append(agent.epsilon)

    print(f"Episode: {episode+1}, Total Reward: {total_reward}, Score: {score}, Epsilon: {agent.epsilon}")

# Lưu Q-table sau khi huấn luyện
q_table_serializable = {str(state): actions for state, actions in agent.q_table.items()}
with open('q_table_montecarlo.json', 'w') as json_file:
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
