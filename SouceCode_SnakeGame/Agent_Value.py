import numpy as np

class ValueIterationAgent:
    def __init__(self, actions=[0, 1, 2], gamma=0.9, theta=0.0001):
        # Khởi tạo V-table để lưu giá trị của trạng thái
        self.v_table = {}  # Dạng {state: value}
        self.policy = {}  # Dạng {state: action}
        self.actions = actions  # Các hành động có thể thực hiện [Giữ nguyên, rẽ phải, rẽ trái]
        self.gamma = gamma  # Hệ số chiết khấu
        self.theta = theta  # Ngưỡng để quyết định dừng quá trình lặp

    def initialize_value(self, state):
        """Khởi tạo giá trị cho trạng thái nếu chưa có"""
        state_tuple = tuple(state)
        if state_tuple not in self.v_table:
            self.v_table[state_tuple] = 0  # Giá trị khởi đầu cho V-table

    def value_iteration(self, env):
        """Thực hiện quá trình Value Iteration để tìm giá trị tối ưu cho từng trạng thái"""
        while True:
            delta = 0
            for state in env.get_all_states():
                state_tuple = tuple(state)
                self.initialize_value(state)

                # Tính giá trị tối ưu của trạng thái dựa trên các hành động có thể thực hiện
                action_values = []
                for action in self.actions:
                    reward, next_state = env.get_reward_and_next_state(state_tuple, action)
                    action_value = reward + self.gamma * self.v_table.get(tuple(next_state), 0)
                    action_values.append(action_value)

                # Chọn giá trị cao nhất từ các hành động
                best_action_value = max(action_values)
                delta = max(delta, abs(self.v_table[state_tuple] - best_action_value))
                self.v_table[state_tuple] = best_action_value

            # Kiểm tra điều kiện hội tụ
            if delta < self.theta:
                break

        # Sau khi có V-table tối ưu, tạo ra chính sách
        self.create_policy(env)

    def create_policy(self, env):
        """Tạo ra chính sách tối ưu từ V-table"""
        for state in env.get_all_states():
            state_tuple = tuple(state)
            action_values = []

            # Tính giá trị Q cho từng hành động
            for action in self.actions:
                reward, next_state = env.get_reward_and_next_state(state_tuple, action)
                action_value = reward + self.gamma * self.v_table.get(tuple(next_state), 0)
                action_values.append(action_value)

            # Chọn hành động tối ưu
            best_action = self.actions[np.argmax(action_values)]
            self.policy[state_tuple] = best_action

    def get_action(self, state):
        """Lấy hành động từ chính sách tối ưu"""
        state_tuple = tuple(state)
        return self.policy.get(state_tuple, np.random.choice(self.actions))
