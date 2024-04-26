import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("Final_Combined_Dataset.csv")

# Remove 'a photograph of' from 'image_captions' column
df['image_captions'] = df['image_captions'].str.replace('a photograph of ', '', regex=False)

# Concatenate 'description' and modified 'image_captions' columns
df['concatenated_desc'] = df['description'] + '\n' + df['image_captions']

# Replace empty strings with empty lists in the 'homeInsights' column
df['homeInsights'] = df['homeInsights'].replace('', '[]')

# Convert string representations of lists into actual lists
def evaluate_list(x):
    try:
        return eval(x)
    except Exception as e:
        return []

df['homeInsights'] = df['homeInsights'].apply(evaluate_list)

for i in range(len(df)):
    if isinstance(df['homeInsights'][i], list) and len(df['homeInsights'][i]) > 0:
        str1 = ''
        for j in range(len(df['homeInsights'][i][0]['insights'])):
            if 'phrases' in df['homeInsights'][i][0]['insights'][j] and len(df['homeInsights'][i][0]['insights'][j]['phrases']) > 0:
                str1 += ' '.join(df['homeInsights'][i][0]['insights'][j]['phrases'])
        df.loc[i, 'concatenated_desc'] = str(df.loc[i, 'concatenated_desc']) + '\n' + str1
    else:
        continue


df.to_csv("Final_Combined_Dataset2.csv", index=False)
