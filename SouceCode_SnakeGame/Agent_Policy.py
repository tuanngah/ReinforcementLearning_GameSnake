import numpy as np

class PolicyIterationAgent:
    def __init__(self, actions=[0, 1, 2], gamma=0.9, theta=0.0001):
        # Khởi tạo V-table để lưu giá trị của trạng thái
        self.v_table = {}  # Dạng {state: value}
        # Chính sách ban đầu
        self.policy = {}  # Dạng {state: action}
        self.actions = actions  # Các hành động có thể thực hiện [Giữ nguyên, rẽ phải, rẽ trái]
        self.gamma = gamma  # Hệ số chiết khấu
        self.theta = theta  # Ngưỡng để quyết định dừng quá trình đánh giá chính sách

    def initialize_policy(self, state):
        """Khởi tạo chính sách ngẫu nhiên cho trạng thái nếu chưa có"""
        state_tuple = tuple(state)
        if state_tuple not in self.policy:
            # Khởi tạo chính sách ngẫu nhiên (chọn hành động ngẫu nhiên)
            self.policy[state_tuple] = np.random.choice(self.actions)
        if state_tuple not in self.v_table:
            self.v_table[state_tuple] = 0  # Giá trị khởi đầu cho V-table

    def policy_evaluation(self, env):
        """Đánh giá chính sách bằng cách cập nhật V-table cho chính sách hiện tại"""
        while True:
            delta = 0
            for state in env.get_all_states():
                state_tuple = tuple(state)
                self.initialize_policy(state)

                # Tính giá trị V(s) dựa trên chính sách hiện tại
                action = self.policy[state_tuple]
                reward, next_state = env.get_reward_and_next_state(state_tuple, action)
                new_v_value = reward + self.gamma * self.v_table.get(tuple(next_state), 0)

                delta = max(delta, abs(self.v_table[state_tuple] - new_v_value))
                self.v_table[state_tuple] = new_v_value

            if delta < self.theta:
                break

    def policy_improvement(self, env):
        """Cải thiện chính sách dựa trên giá trị của V-table"""
        policy_stable = True
        for state in env.get_all_states():
            state_tuple = tuple(state)
            self.initialize_policy(state)

            old_action = self.policy[state_tuple]
            action_values = []

            # Tính giá trị Q cho mỗi hành động tại trạng thái này
            for action in self.actions:
                reward, next_state = env.get_reward_and_next_state(state_tuple, action)
                action_value = reward + self.gamma * self.v_table.get(tuple(next_state), 0)
                action_values.append(action_value)

            # Chọn hành động tốt nhất dựa trên giá trị Q
            best_action = self.actions[np.argmax(action_values)]
            self.policy[state_tuple] = best_action

            if best_action != old_action:
                policy_stable = False

        return policy_stable

    def get_action(self, state):
        """Lấy hành động từ chính sách hiện tại"""
        state_tuple = tuple(state)
        self.initialize_policy(state)
        return self.policy[state_tuple]
