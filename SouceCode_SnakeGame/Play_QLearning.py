from SnakeGame import SnakeGame
from Agent_QLearning import QLearningAgent
import json
import pygame
import time

# Tạo môi trường
game = SnakeGame(render=True)

# Tạo agent và đọc Q-table từ file đã lưu
agent = QLearningAgent(alpha=0.0, gamma=0.0, epsilon=0.0)  # Đặt alpha và epsilon thành 0 vì không cần học thêm nữa

# Đọc file Q-table từ file JSON đã lưu
with open('q_table_100000.json', 'r') as json_file:
    q_table_serializable = json.load(json_file)

# Chuyển Q-table từ string keys về lại tuple keys
agent.q_table = {eval(state): actions for state, actions in q_table_serializable.items()}

# Số lần chơi để quan sát kết quả
episodes = 5
total_scores = 0  # Biến để lưu tổng điểm qua tất cả các lần chơi

# Chơi game với agent đã huấn luyện
for episode in range(episodes):
    state = game.reset()
    game_over = False
    total_reward = 0
    score = 0

    while not game_over:
        # Lựa chọn hành động dựa trên Q-table đã học
        action = agent.get_action(state)
        reward, game_over, score, next_state = game.play_step(action)

        # Cập nhật trạng thái
        state = next_state
        total_reward += reward

        # Thêm thời gian chờ để dễ quan sát
        time.sleep(0.01)  # Điều chỉnh tốc độ chơi

    total_scores += score  # Cộng dồn điểm của mỗi episode
    print(f"Episode: {episode+1}, Score: {score}, Total Reward: {total_reward}")

# Tính và in ra điểm trung bình sau 5 lần chơi
average_score = total_scores / episodes
print(f"Average Score over {episodes} episodes: {average_score}")

# Kết thúc chương trình
pygame.quit()
