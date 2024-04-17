import numpy as np
import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

# Function to generate house preference description
def generate_house_preference(row):
    preference = "This person is looking for a"
    
    if row['Proximity to Hospitals']:
        preference += " house near hospitals,"
    if row['Proximity to Recreational Areas']:
        preference += " close to recreational areas,"
    if row['Proximity to Schools']:
        preference += " with easy access to schools,"
    if row['Proximity to Workplaces']:
        preference += " near workplaces,"
    
    # Additional features
    if np.random.choice([True, False]):
        preference += " sea-facing,"
    if np.random.choice([True, False]):
        preference += " with a garden,"
    if np.random.choice([True, False]):
        preference += " with a swimming pool,"
    if np.random.choice([True, False]):
        preference += " with a garage,"
    
    preference = preference.rstrip(',') + "."
    
    return preference

# Generate synthetic data
num_records = 10000  # Number of records to generate
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

# Add house preference description column
df['House Preference'] = df.apply(generate_house_preference, axis=1)

# Save DataFrame to CSV file
df.to_csv('synthetic_data_with_preferences.csv', index=False)

print("Synthetic data with preferences saved to synthetic_data_with_preferences.csv")
