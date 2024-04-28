import json
file = open("ClusterPoints_Dataset.json", "r")
data = json.load(file)
print(data.keys())