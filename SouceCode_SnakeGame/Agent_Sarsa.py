import numpy as np
import random

class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # Khởi tạo Q-table dưới dạng dictionary
        self.q_table = {}  # Dạng {state: [q-value cho mỗi action]}
        self.alpha = alpha  # Tốc độ học
        self.gamma = gamma  # Hệ số chiết khấu
        self.epsilon = epsilon  # Xác suất khám phá
        self.epsilon_decay = epsilon_decay  # Hệ số giảm epsilon mỗi lần huấn luyện
        self.epsilon_min = epsilon_min  # Giá trị nhỏ nhất của epsilon

    def get_action(self, state):
        """Lấy hành động từ chính sách epsilon-greedy"""
        state_tuple = tuple(state)
        if state_tuple not in self.q_table:
            # Khởi tạo Q-values cho mỗi hành động tại trạng thái mới
            self.q_table[state_tuple] = [0, 0, 0]  # [Giữ nguyên, rẽ phải, rẽ trái]

        if random.uniform(0, 1) < self.epsilon:
            # Khám phá: chọn hành động ngẫu nhiên
            return random.choice([0, 1, 2])  # 0: Giữ nguyên, 1: Rẽ phải, 2: Rẽ trái
        else:
            # Khai thác: chọn hành động có Q-value cao nhất
            return np.argmax(self.q_table[state_tuple])

    def update_q_value(self, state, action, reward, next_state, next_action):
        """Cập nhật giá trị Q-table theo phương trình SARSA"""
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)

        if next_state_tuple not in self.q_table:
            # Khởi tạo Q-values cho trạng thái tiếp theo nếu chưa có
            self.q_table[next_state_tuple] = [0, 0, 0]

        # Giá trị Q hiện tại cho hành động đã chọn
        current_q_value = self.q_table[state_tuple][action]
        # Giá trị Q của hành động tiếp theo theo chính sách hiện tại (on-policy)
        next_q_value = self.q_table[next_state_tuple][next_action]

        # Cập nhật Q theo phương trình SARSA
        new_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * next_q_value)
        self.q_table[state_tuple][action] = new_q_value

    def decay_epsilon(self):
        """Giảm dần epsilon để chuyển từ khám phá sang khai thác"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
