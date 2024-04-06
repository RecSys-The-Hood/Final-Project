import numpy as np
import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

# Generate synthetic data
num_records = 1000  # Number of records to generate
religions = ['Hindu', 'Jain', 'Christian', 'Buddhist', 'Muslim']
data = {
    'Income': [np.random.randint(40000, 200000) for _ in range(num_records)],
    "Name":[fake.name() for _ in range (num_records)],
    'Religion': [np.random.choice((religions),p=[0.2,0.11,0.5,0.09,0.1]) for _ in range(num_records)],
    'Family Size': [np.random.randint(1, 6) for _ in range(num_records)],
    'Age': [np.random.randint(18, 80) for _ in range(num_records)],
    'Marital Status': [np.random.choice(['Bachelor', 'Married'], p=[0.4, 0.6]) for _ in range(num_records)],
    'Proximity to Hospitals': [fake.boolean(chance_of_getting_true=50) for _ in range(num_records)],
    'Proximity to Recreational Areas': [fake.boolean(chance_of_getting_true=50) for _ in range(num_records)],
    'Proximity to Schools': [fake.boolean(chance_of_getting_true=50) for _ in range(num_records)],
    'Proximity to Workplaces': [fake.boolean(chance_of_getting_true=50) for _ in range(num_records)],
    'Property Type': [np.random.choice(['Apartment', 'House', 'Condo', 'Townhouse']) for _ in range(num_records)]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV file
df.to_csv('synthetic_data.csv', index=False)

print("Synthetic data saved to synthetic_data.csv")
