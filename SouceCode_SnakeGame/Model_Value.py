from SnakeGame import SnakeGame
from Agent_Value import ValueIterationAgent
import json
import matplotlib.pyplot as plt

# Tạo môi trường và agent Value Iteration
game = SnakeGame(render=False)
agent = ValueIterationAgent()

# Số lần lặp Value Iteration
iterations = 500

# Khởi tạo danh sách để lưu kết quả của mỗi iteration
rewards = []

# Huấn luyện agent
for iteration in range(iterations):
    # Thực hiện Value Iteration
    agent.value_iteration(game)

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
policy_serializable = {str(state): action for state, action in agent.policy.items()}
v_table_serializable = {str(state): value for state, value in agent.v_table.items()}

# Lưu chính sách sau khi huấn luyện hoàn tất
with open('value_iteration_policy.json', 'w') as json_file:
    json.dump(policy_serializable, json_file, indent=4)

# Lưu V-table sau khi huấn luyện hoàn tất
with open('value_iteration_v_table.json', 'w') as json_file:
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
