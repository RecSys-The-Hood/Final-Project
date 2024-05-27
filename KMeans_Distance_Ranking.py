
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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import json

def filter_by_labels(data, labels):
    # Use list comprehension to get dictionaries with matching labels
    filtered_data = [entry for entry in data if entry['labels'] == labels]
    return filtered_data


users = pd.read_csv("Users.csv")
results = pd.DataFrame(columns=["Mean ROUGE Score", "Variance in ROUGE Score"])

cols = ['bedrooms', 'bathrooms', 'price', 'livingArea', 'leisure_within_5km', 'shops_within_5km', 'schools_within_5km', 'transit_within_2km', 'homeType']
# Define the URL of the backend server
server_url = "http://127.0.0.1:5000/predict"  

df=pd.read_csv("./Final_Combined_Dataset.csv")
df=df.set_index(keys="zpid")

df1=df.drop(columns=['zipcode','hdpUrl','cityId','livingAreaValue','rentZestimate','photoCount','address.streetAddress','originalPhotos','latitude','longitude'])

null_percentage = (df1.isnull().sum() / len(df1)) * 100

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


label_encoder = LabelEncoder()
df1['homeType'] = label_encoder.fit_transform(df1['homeType'])
encoded_classes = label_encoder.classes_
print(encoded_classes)

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


for index, row in users.iterrows():

    state = row['state']
    weights = [row['weight_bed'],row['weight_bath'],row['weight_price'],row['weight_area'],
               row['weight_leisure'],row['weight_shop'],row['weight_school'],
               row['weight_transit'],row['weight_type']]
    
    column = 'description'

    # predict_X=[2,1,1700000,2000,20,400,250,200,2]  # give input here
    predict_X = [row['val_bed'],row['val_bath'],row['val_price'],row['val_area'],
               row['val_leisure'],row['val_shop'],row['val_school'],
               row['val_transit'],row['val_type']]
    
    description = row['description']    


    dfs_by_category_final = {}
    dfs_cluster = {}

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
            
            X=df_result.values
            X=X*weights          
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

    homeType_Encoding= {
        'APARTMENT': 0,
        'CONDO': 1,
        'MANUFACTURED': 2,
        'MULTI_FAMILY': 3,
        'SINGLE_FAMILY': 4,
        'TOWNHOUSE': 5
    }

    kmeans = joblib.load(f"{state}_kmeans.pkl")
    scaler = joblib.load(f"{state}_scaler.pkl")
    
    json_file_path = "./ClusterPoints_Dataset.json"

    with open(json_file_path, "r") as file:
        kmeansdata = json.load(file)

    json_file_path_1 = "./Labelled_Dataset_with_zpid.json"

    with open(json_file_path_1, "r") as file:
        fulldata = json.load(file)

    df_predict = pd.DataFrame([predict_X],columns=cols)
    scaler_predict=StandardScaler()

    df_to_transform_p = df_predict.drop(columns=['homeType'])
    df_excluded_p = df_predict[['homeType']]
    # Apply the transformation
    transformed_array_p = scaler.transform(df_to_transform_p)
    df_transformed_p = pd.DataFrame(transformed_array_p, columns=df_to_transform_p.columns, index=df_predict.index)
    # Recombine the DataFrame
    df_result_p = pd.concat([df_transformed_p, df_excluded_p], axis=1)
    # print(df_result)
    X_p=df_result_p.values
    X_p=X_p*weights
    
    predicted_label=kmeans.predict(X_p)
    state_full_data=fulldata[state]

    state_filtered_data = filter_by_labels(state_full_data, predicted_label[0])
    
    df_labels = pd.DataFrame(state_filtered_data)
    df_labels = df_labels.set_index(keys=['zpid'])
    
    df_data = df_labels.drop(columns=['labels'])
    df_home= df_data['homeType']
    df_data=df_data.drop(columns=['homeType'])
    df_final = pd.concat((df_data, df_home),axis=1)

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
    
    cluster_centers = kmeans.cluster_centers_
    cluster_of_interest = cluster_centers[predicted_label[0]]
    
    X = np.array(X)
    Y = np.array(cluster_of_interest)

    # Calculate Euclidean distances
    distances = np.linalg.norm(X - Y, axis=1)

    # Get the indices that would sort the distances
    sorted_indices = np.argsort(distances)
    row_indexes = df_final.index[sorted_indices]
    # Sort the 2D matrix based on the distances
    sorted_X = X[sorted_indices]
    
    sortedX = sorted_X/weights
    X_to_denorm = sortedX[:,:-1]
    # Denormalize the cluster centers
    X_denormalized = scaler.inverse_transform(X_to_denorm)

    X_final= np.concatenate((X_denormalized,sortedX[:,-1].reshape(-1, 1)),axis=1)
    
    df_X=pd.DataFrame(X_final,columns=list(cols))
    df_sorted=np.floor(df_X)
    df_sorted=df_sorted.set_index(row_indexes)
    
    top_10_recommendations = df_sorted.head(10)
    top_10_recommendations = top_10_recommendations.reset_index()
    df_summary = pd.read_csv("combined_summary_data.csv")
    top_10_recommendations = pd.merge(top_10_recommendations, df_summary, on='zpid', how='inner')

    rouge_scorer = Rouge()

    rouge_scores = [rouge_scorer.get_scores(
        hyps=description,
        refs=reference,
    )[0]["rouge-l"]["f"] for reference in top_10_recommendations[column].tolist()]

    dict = {
        "Mean ROUGE Score" : statistics.mean(rouge_scores), 
        "Variance in ROUGE Score" : statistics.variance(rouge_scores)
    }

    results.loc[len(results)] = dict

results.to_csv("Results_With_Cluster_centers.csv", index=False)
print("Done")
