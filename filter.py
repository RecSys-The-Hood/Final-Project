import pandas as pd

# Load the CSV data
df = pd.read_csv('./Final_Combined_Dataset.csv')

# Create a new column that concatenates 'description' and 'image_captions'
df['Concat'] = df['description'] + df['image_captions']

# Create a new DataFrame with rows where the length of 'Concat' is less than 1024
df1 = df[df['Concat'].str.len() >=1024]

# Save the new DataFrame to a CSV file
df1.to_csv('Final_dataset_big.csv', index=False)

print(df1.shape)