import csv
import random
import json
import math
import pandas as pd

# Define the states
states = ["GA", "IL", "TX", "MI", "CA", "FL", "NY", "PA", "AZ", "CA", "DC"]

df = pd.read_csv("Descriptions.csv", header=None)
df.columns = ['Desc']

with open('rates.json', 'r') as json_file:
    rates = json.load(json_file)

with open('schools.json', 'r') as json_file:
    schools = json.load(json_file)

with open('transit.json', 'r') as json_file:
    transit = json.load(json_file)

with open('shops.json', 'r') as json_file:
    shops = json.load(json_file)

with open('leisure.json', 'r') as json_file:
    leisure = json.load(json_file)

with open('schools_std.json', 'r') as json_file:
    schools_std = json.load(json_file)

with open('transit_std.json', 'r') as json_file:
    transit_std = json.load(json_file)

with open('shops_std.json', 'r') as json_file:
    shops_std = json.load(json_file)

with open('leisure_std.json', 'r') as json_file:
    leisure_std = json.load(json_file)

# Function to generate random weights of either 1 or 5, with `weight_type` as 1
def generate_weights():
    weights = [1] * 6 + [5] * 2  # Create a list with exactly 7 ones and 2 fives
    random.shuffle(weights)      # Shuffle the list to randomize the order
    weights.append(1)
    return weights

# Generate synthetic data
data = []
for i in range(50):
    state = random.choice(states)
    weights = generate_weights()
    
    # Ensure fewer houses with more than 3 bedrooms or bathrooms
    if random.random() < 0.8:  # 80% chance to have 3 or fewer bedrooms/bathrooms
        val_bed = random.randint(1, 3)
        val_bath = random.randint(1, 3)
    else:
        val_bed = random.randint(4, 6)
        val_bath = random.randint(4, 6)
    
    # val_price = random.randint(200000, 3000000)
    val_area = random.randint(800, 5000)

    temp_p = val_area*rates[state] 
    val_price = math.ceil(temp_p) + random.randint(-1*math.ceil(0.1*temp_p), math.ceil(0.1*temp_p))
    val_leisure = math.ceil(leisure[state]) + random.randint(-1*math.ceil(leisure_std[state]), math.ceil(leisure_std[state]))
    val_shop = math.ceil(shops[state]) + random.randint(-1*math.ceil(shops_std[state]), math.ceil(shops_std[state]))
    val_school = math.ceil(schools[state]) + random.randint(-1*math.ceil(schools_std[state]), math.ceil(schools_std[state]))
    val_transit = math.ceil(transit[state]) + random.randint(-1*math.ceil(transit_std[state]), math.ceil(transit_std[state]))
    val_type = 2  # House type value is always 2
    
    # Generate a house-related description without mentioning number of bedrooms and bathrooms
    description = df.iloc[i]['Desc']
    
    data.append([
        *weights, val_bed, val_bath, val_price, val_area, val_leisure,
        val_shop, val_school, val_transit, val_type, description, state
    ])

# Write to CSV
with open('Users_Synthetic.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "weight_bed", "weight_bath", "weight_price", "weight_area",
        "weight_leisure", "weight_shop", "weight_school", "weight_transit",
        "weight_type", "val_bed", "val_bath", "val_price", "val_area",
        "val_leisure", "val_shop", "val_school", "val_transit", "val_type",
        "description", "state"
    ])
    writer.writerows(data)
