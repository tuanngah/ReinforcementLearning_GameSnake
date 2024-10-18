from SnakeGame import SnakeGame
from Agent_Policy import PolicyIterationAgent
import json
import matplotlib.pyplot as plt

# Tạo môi trường và agent Policy Iteration
game = SnakeGame(render=False)
agent = PolicyIterationAgent()

# Số lượng iterations
iterations = 500

# Khởi tạo danh sách để lưu kết quả
rewards = []

# Huấn luyện agent
for iteration in range(iterations):
    # Bước 1: Đánh giá chính sách hiện tại
    agent.policy_evaluation(game)

    # Bước 2: Cải thiện chính sách
    policy_stable = agent.policy_improvement(game)

    if policy_stable:
        print(f"Chính sách đã ổn định sau {iteration + 1} iterations.")
        break

    # Chơi thử game sau mỗi iteration để xem kết quả
    state = game.reset()
    total_reward = 0
    game_over = False

    while not game_over:
        action = agent.get_action(state)
        reward, game_over, score, next_state = game.play_step(action)
        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    print(f"Iteration: {iteration+1}, Total Reward: {total_reward}, Score: {score}")

# Lưu chính sách và V-table sau khi huấn luyện
policy_serializable = {str(state): int(action) for state, action in agent.policy.items()}
v_table_serializable = {str(state): float(value) for state, value in agent.v_table.items()}

with open('policy_iteration_policy.json', 'w') as json_file:
    json.dump(policy_serializable, json_file, indent=4)

with open('policy_iteration_v_table.json', 'w') as json_file:
    json.dump(v_table_serializable, json_file, indent=4)

print("Training completed and policy saved!")

# Vẽ biểu đồ kết quả huấn luyện
def plot_training_results(rewards):
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Iteration')
    plt.tight_layout()
    plt.show()

# Vẽ biểu đồ quá trình huấn luyện
plot_training_results(rewards)
