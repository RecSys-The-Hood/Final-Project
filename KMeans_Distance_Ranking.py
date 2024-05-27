# %%
import random

# from sentence_transformers import SentenceTransformer
import pandas as pd
# import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import joblib
import requests

from sklearn.cluster import KMeans

from sacrebleu.metrics import BLEU
from rouge import Rouge
import statistics

# %% [markdown]
# make sure server is on and the pkl files of kmeans clustes is available.

df=pd.read_csv("./Final_Combined_Dataset.csv")
df=df.set_index(keys="zpid")
# df

# %%
# df.columns

# %%
df1=df.drop(columns=['zipcode','hdpUrl','cityId','livingAreaValue','rentZestimate','photoCount','address.streetAddress','originalPhotos','latitude','longitude'])

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
unique_categories=unique_categories[:-1]
print(unique_categories)
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
        df_to_transform = df_check.drop(columns=['homeType'])
        df_excluded = df_check[['homeType']]
        # Apply the transformation
        scaler = StandardScaler()
        transformed_array = scaler.fit_transform(df_to_transform)
        df_transformed = pd.DataFrame(transformed_array, columns=df_to_transform.columns, index=df_check.index)
        # Recombine the DataFrame
        df_result = pd.concat([df_transformed, df_excluded], axis=1)
        # print(df_result)
        X=df_result.values
        weights = [5,5,1,1,1,1,1,1,1]
        X=X*weights
                # Try different values of k (number of clusters)
        # for k in range(2, 40):            
        kmeans = KMeans(n_clusters=12,init='k-means++', random_state=42)
        if(key=='CA'):
            kmeans = KMeans(n_clusters=24,init='k-means++', random_state=42)

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
        joblib.dump(scaler,f'{key}_scaler.pkl')

# Convert to dictionary format for JSON serialization
json_data = {key: df.to_dict(orient='records') for key, df in dfs_by_category_final.items()}

# Ensure conversion to Python-native types
json_cluster_data = {key: value.tolist() for key, value in dfs_cluster.items()}

# Save to JSON files
with open('Labelled_Dataset_with_zpid.json', 'w') as f:
    json.dump(json_data, f, indent=4)

with open('ClusterPoints_Dataset.json', 'w') as f:
    json.dump(json_cluster_data, f, indent=4)

# %%
weights = [5,5,1,1,1,1,1,1,1]
cols = ['bedrooms', 'bathrooms', 'price', 'livingArea', 'leisure_within_5km', 'shops_within_5km', 'schools_within_5km', 'transit_within_2km', 'homeType']

# %% [markdown]
# Running on Sample Input

# %%
state = 'NY'
kmeans = joblib.load(f"{state}_kmeans.pkl")
scaler = joblib.load(f"{state}_scaler.pkl")

# %%
cluster_centers = kmeans.cluster_centers_

# Since we applied weights, we need to reverse this before denormalization
cluster_centers_unweighted = cluster_centers /weights

cluster_centers_to_denorm = cluster_centers_unweighted[:,:-1]
# print(cluster_centers_to_denorm)
# Denormalize the cluster centers
cluster_centers_denormalized = scaler.inverse_transform(cluster_centers_to_denorm)

cluster_centers_final= np.concatenate((cluster_centers_denormalized,cluster_centers_unweighted[:,-1].reshape(-1, 1)),axis=1)
cluster_centers_final

# %%
df_cluster_centers=pd.DataFrame(cluster_centers_final,columns=list(cols))
np.floor(df_cluster_centers)

# %%
predict_X=[2,1,1700000,2000,20,400,250,200,2]  # give input here
description = "A large backyard, swimming pool, spacious living room, furniture in bedroom, tiled bathroom"
df_predict = pd.DataFrame([predict_X],columns=cols)
scaler_predict=StandardScaler()

df_to_transform_p = df_predict.drop(columns=['homeType'])
df_excluded_p = df_predict[['homeType']]
# Apply the transformation
transformed_array_p = scaler.transform(df_to_transform_p)
print(transformed_array_p)
df_transformed_p = pd.DataFrame(transformed_array_p, columns=df_to_transform_p.columns, index=df_predict.index)
# Recombine the DataFrame
df_result_p = pd.concat([df_transformed_p, df_excluded_p], axis=1)
# print(df_result)
X_p=df_result_p.values
X_p=X_p*weights
X_p

# %%
kmeans.predict(X_p)

# %% [markdown]
# Running on Server Response

# %%
homeType_Encoding= {
        'APARTMENT': 0,
        'CONDO': 1,
        'MANUFACTURED': 2,
        'MULTI_FAMILY': 3,
        'SINGLE_FAMILY': 4,
        'TOWNHOUSE': 5
    }

# %%
state = 'NY'
kmeans = joblib.load(f"{state}_kmeans.pkl")
scaler = joblib.load(f"{state}_scaler.pkl")
weights = [5,5,1,1,1,1,1,1,1]


# %%
json_file_path = "./ClusterPoints_Dataset.json"

with open(json_file_path, "r") as file:
    kmeansdata = json.load(file)

json_file_path_1 = "./Labelled_Dataset_with_zpid.json"

with open(json_file_path_1, "r") as file:
    fulldata = json.load(file)

# %%
predict_X=[2,1,1700000,2000,20,400,250,200,2]  # give input here
df_predict = pd.DataFrame([predict_X],columns=cols)
scaler_predict=StandardScaler()

df_to_transform_p = df_predict.drop(columns=['homeType'])
df_excluded_p = df_predict[['homeType']]
# Apply the transformation
transformed_array_p = scaler.transform(df_to_transform_p)
print(transformed_array_p)
df_transformed_p = pd.DataFrame(transformed_array_p, columns=df_to_transform_p.columns, index=df_predict.index)
# Recombine the DataFrame
df_result_p = pd.concat([df_transformed_p, df_excluded_p], axis=1)
# print(df_result)
X_p=df_result_p.values
X_p=X_p*weights
X_p

# %%
predicted_label=kmeans.predict(X_p)
predicted_label

# %%
state_full_data=fulldata[state]
state_full_data

# %%
def filter_by_labels(data, labels):
    # Use list comprehension to get dictionaries with matching labels
    filtered_data = [entry for entry in data if entry['labels'] == labels]
    return filtered_data

# %%
state_filtered_data = filter_by_labels(state_full_data, predicted_label[0])
len(state_filtered_data)

# %%
state_filtered_data

# %% [markdown]
# Sorting Entries based on distance from cluster

# %%
df = pd.DataFrame(state_filtered_data)
df=df.set_index(keys=['zpid'])
df

# %%
df_data = df.drop(columns=['labels'])
df_home= df_data['homeType']
df_data=df_data.drop(columns=['homeType'])
df_final = pd.concat((df_data, df_home),axis=1)
df_final

# %%
np.unique(df_final['bathrooms'])

# %%
df_to_transform = df_final.drop(columns=['homeType'])
df_excluded= df_final[['homeType']]
# Apply the transformation
transformed_array = scaler.transform(df_to_transform)
df_transformed = pd.DataFrame(transformed_array, columns=df_to_transform.columns, index=df_final.index)
# Recombine the DataFrame
df_result= pd.concat([df_transformed, df_excluded], axis=1)
# print(df_result)
X=df_result.values
X=X*weights
X[:,1]

# %%
cluster_centers = kmeans.cluster_centers_
cluster_of_interest = cluster_centers[predicted_label[0]]
cluster_of_interest

# %%
X = np.array(X)

Y = np.array(cluster_of_interest)

# Calculate Euclidean distances
distances = np.linalg.norm(X - Y, axis=1)

# Get the indices that would sort the distances
sorted_indices = np.argsort(distances)
row_indexes = df_final.index[sorted_indices]
# Sort the 2D matrix based on the distances
sorted_X = X[sorted_indices]


# %%
print("\nSorted matrix:")
print(sorted_X[:,1])

# %%
sortedX = sorted_X /weights

X_to_denorm = sortedX[:,:-1]
# print(cluster_centers_to_denorm)
# Denormalize the cluster centers
X_denormalized = scaler.inverse_transform(X_to_denorm)

X_final= np.concatenate((X_denormalized,sortedX[:,-1].reshape(-1, 1)),axis=1)
X_final

# %%
df_X=pd.DataFrame(X_final,columns=list(cols))
df_sorted=np.floor(df_X)
df_sorted=df_sorted.set_index(row_indexes)
df_sorted

# %%
top_10_recommendations = df_sorted.head(10)
top_10_recommendations = top_10_recommendations.reset_index()
top_10_recommendations

# %%
df = pd.read_csv("combined_summary_data.csv")

top_10_recommendations = pd.merge(top_10_recommendations, df, on='zpid', how='inner')

# %%
bleu_scorer = BLEU()

column = 'summary' ## Change the column name as needed,, make it 'summary' for the best model

bleu_scores = [bleu_scorer.sentence_score(
    hypothesis=description,
    references=[reference],
).score for reference in top_10_recommendations[column].tolist()]

bleu_scores

# %%
print(statistics.mean(bleu_scores))
print(statistics.variance(bleu_scores))

# %%
rouge_scorer = Rouge()

rouge_scores = [rouge_scorer.get_scores(
    hyps=description,
    refs=reference,
)[0]["rouge-l"]["f"] for reference in top_10_recommendations[column].tolist()]
# score[0]["rouge-l"]["f"]

rouge_scores

# %%
print(statistics.mean(rouge_scores))
print(statistics.variance(rouge_scores))


