import json
from SnakeGame import SnakeGame
from Agent_Policy import PolicyIterationAgent
import pygame
import time

# Tạo môi trường game Snake
game = SnakeGame(render=True)  # Bật render để quan sát agent chơi

# Tạo agent Policy Iteration và đọc chính sách từ file đã lưu
agent = PolicyIterationAgent()

# Đọc chính sách và V-table đã huấn luyện từ file
with open('policy_iteration_policy.json', 'r') as json_file:
    policy_serializable = json.load(json_file)
agent.policy = {eval(state): action for state, action in policy_serializable.items()}

with open('policy_iteration_v_table.json', 'r') as json_file:
    v_table_serializable = json.load(json_file)
agent.v_table = {eval(state): value for state, value in v_table_serializable.items()}

# Số lần chơi để quan sát kết quả
episodes = 5

# Chơi game với agent đã huấn luyện
for episode in range(episodes):
    state = game.reset()
    game_over = False
    total_reward = 0

    while not game_over:
        # Lựa chọn hành động dựa trên chính sách đã học
        action = agent.get_action(state)
        reward, game_over, score, next_state = game.play_step(action)

        # Cập nhật trạng thái
        state = next_state
        total_reward += reward

        # Thêm thời gian chờ để dễ quan sát
        time.sleep(0.1)  # Điều chỉnh tốc độ chơi

    print(f"Episode: {episode+1}, Score: {score}, Total Reward: {total_reward}")

# Kết thúc chương trình
pygame.quit()
