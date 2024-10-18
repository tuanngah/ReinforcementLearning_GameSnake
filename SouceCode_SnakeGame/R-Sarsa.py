import json

# Đọc file Q-table của SARSA
with open('q_table_sarsa.json', 'r') as json_file:
    sarsa_table_serializable = json.load(json_file)

# Chuyển Q-table từ string keys về lại tuple keys
sarsa_table = {eval(state): actions for state, actions in sarsa_table_serializable.items()}

# In ra bảng Q-table (nếu cần thiết)
print("SARSA Q-table loaded successfully!")
