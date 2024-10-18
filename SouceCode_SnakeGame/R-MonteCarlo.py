import json

# Đọc file Q-table của Monte Carlo
with open('q_table_montecarlo.json', 'r') as json_file:
    montecarlo_table_serializable = json.load(json_file)

# Chuyển Q-table từ string keys về lại tuple keys
montecarlo_table = {eval(state): actions for state, actions in montecarlo_table_serializable.items()}

# In ra bảng Q-table (nếu cần thiết)
print("Monte Carlo Q-table loaded successfully!")
