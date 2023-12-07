import json

params = {'learning_rate': 0.001, 'epochs': 10, 'batch_size': 32}

params_file_path = './tmp/params.json'
with open(params_file_path, 'w') as json_file:
    json.dump(params, json_file)

with open(params_file_path, 'r') as json_file:
    loaded_params = json.load(json_file)

print("Loaded Parameters:", loaded_params)
print("Loaded Parameters:", loaded_params['learning_rate'])
