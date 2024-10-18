import json

# Đọc file chính sách của Policy Iteration
with open('policy_iteration_policy.json', 'r') as json_file:
    policy_table_policy_serializable = json.load(json_file)

# Chuyển policy từ string keys về lại tuple keys
policy_table_policy = {eval(state): action for state, action in policy_table_policy_serializable.items()}

# Đọc file V-table của Policy Iteration
with open('policy_iteration_v_table.json', 'r') as json_file:
    policy_table_v_table_serializable = json.load(json_file)

# Chuyển V-table từ string keys về lại tuple keys
policy_table_v_table = {eval(state): value for state, value in policy_table_v_table_serializable.items()}

# In ra bảng chính sách và V-table (nếu cần thiết)
print("Policy Iteration policy and V-table loaded successfully!")
