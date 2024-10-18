import numpy as np
import random

class MonteCarloAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # Khởi tạo Q-table dưới dạng dictionary
        self.q_table = {}  # Dạng {state: [q-value cho mỗi action]}
        self.alpha = alpha  # Tốc độ học
        self.gamma = gamma  # Hệ số chiết khấu
        self.epsilon = epsilon  # Xác suất khám phá
        self.epsilon_decay = epsilon_decay  # Hệ số giảm epsilon
        self.epsilon_min = epsilon_min  # Giá trị nhỏ nhất của epsilon
        self.episode_memory = []  # Lưu trữ toàn bộ episode gồm (state, action, reward)

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

    def store_episode(self, state, action, reward):
        """Lưu trữ episode với (state, action, reward)"""
        self.episode_memory.append((state, action, reward))

    def update_q_values(self):
        """Cập nhật giá trị Q-table sau khi episode kết thúc"""
        G = 0  # Giá trị tổng phần thưởng tích lũy
        for state, action, reward in reversed(self.episode_memory):
            G = self.gamma * G + reward  # Tính toán phần thưởng tích lũy từ tương lai
            state_tuple = tuple(state)

            # Khởi tạo Q-values nếu chưa có
            if state_tuple not in self.q_table:
                self.q_table[state_tuple] = [0, 0, 0]

            # Cập nhật giá trị Q của hành động tại trạng thái này
            old_q_value = self.q_table[state_tuple][action]
            new_q_value = old_q_value + self.alpha * (G - old_q_value)
            self.q_table[state_tuple][action] = new_q_value

        # Xóa bộ nhớ của episode sau khi cập nhật
        self.episode_memory = []

    def decay_epsilon(self):
        """Giảm dần epsilon để giảm mức độ khám phá theo thời gian"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
