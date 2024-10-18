import numpy as np
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # Khởi tạo Q-table dưới dạng dictionary
        self.q_table = {}  # Dạng {state: [q-value cho mỗi action]}
        self.alpha = alpha  # Tốc độ học
        self.gamma = gamma  # Hệ số chiết khấu
        self.epsilon = epsilon  # Xác suất khám phá
        self.epsilon_decay = epsilon_decay  # Hệ số giảm epsilon mỗi lần huấn luyện
        self.epsilon_min = epsilon_min  # Giá trị nhỏ nhất của epsilon

    def get_action(self, state):
        state_tuple = tuple(state)
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = [0, 0, 0]  # Khởi tạo Q-values cho state
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2])  # Khám phá
        else:
            return np.argmax(self.q_table[state_tuple])  # Khai thác

    def update_q_value(self, state, action, reward, next_state):
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)

        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = [0, 0, 0]

        current_q_value = self.q_table[state_tuple][action]
        max_future_q_value = max(self.q_table[next_state_tuple])
        new_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_future_q_value)
        self.q_table[state_tuple][action] = new_q_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
