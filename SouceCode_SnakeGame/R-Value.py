import json

# Đọc file chính sách của Value Iteration
with open('value_iteration_policy.json', 'r') as json_file:
    value_table_policy_serializable = json.load(json_file)

# Chuyển policy từ string keys về lại tuple keys
value_table_policy = {eval(state): action for state, action in value_table_policy_serializable.items()}

# Đọc file V-table của Value Iteration
with open('value_iteration_v_table.json', 'r') as json_file:
    value_table_v_table_serializable = json.load(json_file)

# Chuyển V-table từ string keys về lại tuple keys
value_table_v_table = {eval(state): value for state, value in value_table_v_table_serializable.items()}

# In ra bảng chính sách và V-table (nếu cần thiết)
print("Value Iteration policy and V-table loaded successfully!")
