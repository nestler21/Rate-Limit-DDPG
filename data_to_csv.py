import os
import json
import pandas as pd


# Script used for creating a csv export from the data.json files

folder_path = 'data/Runde 13/'

params_list = ['alpha_a', 'alpha_c', 'l2_value', 'gamma', 'buffer_size', 'tau', 'fc1', 'fc2', 'batch_size', 'noise', 'maxdelay', 'statescope', 'testserver', 'simulation']
runs = []
params = []
values = []
successful_requests = []
time_spent = []
rates= []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            dict = json.load(file)

        runs.append(file_name.split("_")[0])
        for key in params_list.keys():
            if key in file_name:
                params.append(key)
                break
        values.append(file_name.replace(".json", "").split("_")[-1])
        successful_requests.append(len([r for r in dict["rewards"] if r != -1]))
        time_spent.append(sum(dict["actions"]))
        rates.append(successful_requests[-1]/time_spent[-1])

data = {
    "Parameter": params,
    "Value": values,
    "Run": runs,
    "Succesful Requests": successful_requests,
    "Time Spent": time_spent,
    "Succesful Requests per Second": rates
}

df = pd.DataFrame(data)
df.to_csv(folder_path+"data.csv", index=False)
print(df)