# %%
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib
# %%
df=pd.read_csv("./Final_Combined_Dataset.csv")
df=df.set_index(keys="zpid")
# df

# %%
# df.columns

# %%
df1=df.drop(columns=['zipcode','hdpUrl','cityId','livingAreaValue','rentZestimate','photoCount','address.streetAddress','originalPhotos','latitude','longitude','concatenated_desc'])

# df1

# %%
null_percentage = (df1.isnull().sum() / len(df1)) * 100
# null_percentage

# %%
# df1['leisure_within_5km'].median(),df1['transit_within_2km'].median(),df1['schools_within_5km'].median(),df1['transit_within_2km'].median()


# %%
mode_value = df1['bedrooms'].mode()[0]
mod_1=df1['bathrooms'].mode()[0]
living_mod=df1['livingArea'].mean()
# Replace null values with the mode
df1['bedrooms']=df1['bedrooms'].fillna(mode_value)
df1['bathrooms']=df1['bathrooms'].fillna(mod_1)
df1['livingArea']=df1['livingArea'].fillna(living_mod)

columns_to_check = ['description', 'homeInsights','image_captions']

# Remove rows with all null values in specified columns
df1 = df1.dropna(subset=columns_to_check, how='all')

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


label_encoder = LabelEncoder()
df1['homeType'] = label_encoder.fit_transform(df1['homeType'])
encoded_classes = label_encoder.classes_
print(encoded_classes)
# %%
dfs_by_category = {}

# Get unique values in the 'Category' column
unique_categories = df1['address.state'].unique()

# Iterate over unique categories
for category in unique_categories:
    # Filter the DataFrame for the current category
    filtered_df = df1[df1['address.state'] == category]
    
    # Add the filtered DataFrame to the dictionary with category as the key
    dfs_by_category[category] = filtered_df

# Access the DataFrames using category keys
for category, df_category in dfs_by_category.items():
    print(f"Category: {category}")
    print(len(df_category))
    # print(df_category)

# %%
# MSE= {}
# for key,value in dfs_by_category.items():
#     if(len(value)>20):
#         df_check=value
#         df_check=df_check.drop(columns=['description','homeInsights','image_captions','address.city','address.state'])
#         scaler=StandardScaler()
#         X=scaler.fit_transform(df_check)
#         sse = []
#         # Try different values of k (number of clusters)
#         for k in range(2, 40):
#             kmeans = KMeans(n_clusters=k,init='k-means++', random_state=42)
#             kmeans.fit(X)
#             sse.append(kmeans.inertia_) 
        
#         MSE[key]=sse

# %%
# MSE

# %%
# for key,value in MSE.items():
        
#     plt.plot(range(2, 40), value, marker='o')
#     plt.title(f'Elbow Method: {key}')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Sum of Squared Distances')
#     plt.show()

# %%
dfs_by_category_final = {}
dfs_cluster = {}
import json
# Assume dfs_by_category is already defined
for key, value in dfs_by_category.items():
    if len(value) > 20:
        df_check = value
        
        # Keep only essential columns
        df_check = df_check.drop(columns=[
            'description', 'homeInsights', 'image_captions', 
            'address.city', 'address.state', 'address.zipcode'
        ])

        # Standardize the data and apply KMeans clustering
        scaler = StandardScaler()
        X = scaler.fit_transform(df_check)
        kmeans = KMeans(n_clusters=12, init='k-means++', random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_

        # Add labels to the DataFrame
        df_check['labels'] = labels

        # Reset the index to ensure the original index (like 'zpid') becomes a column
        df_check = df_check.reset_index()

        # Convert columns to JSON-serializable types
        df_check = df_check.astype({
            'zpid': int,
            'bedrooms': int,
            'bathrooms': int,
            'price': float,
            'livingArea': float,
            'labels': int
        })

        dfs_by_category_final[key] = df_check
        dfs_cluster[key] = kmeans.cluster_centers_

        # Save KMeans model
        joblib.dump(kmeans, f'{key}_kmeans.pkl')

# Convert to dictionary format for JSON serialization
json_data = {key: df.to_dict(orient='records') for key, df in dfs_by_category_final.items()}

# Ensure conversion to Python-native types
json_cluster_data = {key: value.tolist() for key, value in dfs_cluster.items()}

# Save to JSON files
with open('Labelled_Dataset_with_zpid.json', 'w') as f:
    json.dump(json_data, f, indent=4)

with open('ClusterPoints_Dataset.json', 'w') as f:
    json.dump(json_cluster_data, f, indent=4)
# df_combined = pd.concat(dfs_by_category_final.values())

# # Save the combined dataframe to a CSV file
# df_combined.to_csv("all_dataframes.csv", index=False)

# %%
# cluster_centers=scaler.inverse_transform(kmeans.cluster_centers_)
# cols=df_check.columns
# df_cluster_centers=pd.DataFrame(cluster_centers,columns=list(cols))
# np.ceil(df_cluster_centers)


