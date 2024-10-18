import json

# Đọc file Q-table
with open('q_table.json', 'r') as json_file:
    q_table_serializable = json.load(json_file)

# Chuyển Q-table từ string keys về lại tuple keys
q_table = {eval(state): actions for state, actions in q_table_serializable.items()}

# In ra bảng Q-table (nếu cần thiết)
print("Q-table loaded successfully!")
