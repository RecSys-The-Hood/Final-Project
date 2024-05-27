import csv
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Define the states
states = ["GA", "MA", "IL", "TX", "MI", "CA", "FL", "NY", "PA", "AZ", "CA", "CA", "DC"]

# Function to generate random weights of either 1 or 5, with `weight_type` as 1
def generate_weights():
    return [random.choice([1, 5]) if i < 8 else 1 for i in range(9)]

# Generate synthetic data
data = []
for _ in range(50):
    state = random.choice(states)
    weights = generate_weights()
    
    # Ensure fewer houses with more than 3 bedrooms or bathrooms
    if random.random() < 0.8:  # 80% chance to have 3 or fewer bedrooms/bathrooms
        val_bed = random.randint(1, 3)
        val_bath = random.randint(1, 3)
    else:
        val_bed = random.randint(4, 6)
        val_bath = random.randint(4, 6)
    
    val_price = random.randint(200000, 3000000)
    val_area = random.randint(800, 5000)
    val_leisure = random.randint(10, 100)
    val_shop = random.randint(50, 500)
    val_school = random.randint(100, 400)
    val_transit = random.randint(50, 300)
    val_type = 2  # House type value is always 2
    
    # Generate a house-related description without mentioning number of bedrooms and bathrooms
    description = (
        f"A {random.choice(['spacious', 'cozy', 'modern', 'luxurious'])} house, "
        f"featuring a {random.choice(['large backyard', 'swimming pool', 'spacious living room', 'modern kitchen'])}. "
        f"Located in a prime area, this property is perfect for families looking for "
        f"{random.choice(['convenience', 'comfort', 'luxury', 'a quiet neighborhood'])}."
    )
    
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
