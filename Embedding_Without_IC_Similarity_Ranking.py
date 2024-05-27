import pandas as pd
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



# global vars
users = pd.read_csv("Users.csv")

results = pd.DataFrame(columns=["Mean ROUGE Score", "Variance in ROUGE Score"])

cols = ['bedrooms', 'bathrooms', 'price', 'livingArea', 'leisure_within_5km', 'shops_within_5km', 'schools_within_5km', 'transit_within_2km', 'homeType']
# Define the URL of the backend server
server_url = "http://127.0.0.1:5000/predict"  # Replace with your server URL

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

print("Unique categories")
print(unique_categories)
# Iterate over unique categories
for category in unique_categories:
    # Filter the DataFrame for the current category
    filtered_df = df1[df1['address.state'] == category]
    
    # Add the filtered DataFrame to the dictionary with category as the key
    dfs_by_category[category] = filtered_df




for index, row in users.iterrows():

    # user variables
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
    # clustering code
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
            # print(df_result)
            X=df_result.values
            # weights = [5,5,1,1,1,1,1,1,1]
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
    json_data = {key: df_cat.to_dict(orient='records') for key, df_cat in dfs_by_category_final.items()}

    # Ensure conversion to Python-native types
    json_cluster_data = {key: value.tolist() for key, value in dfs_cluster.items()}

    # Save to JSON files
    with open('Labelled_Dataset_with_zpid.json', 'w') as f:
        json.dump(json_data, f, indent=4)

    with open('ClusterPoints_Dataset.json', 'w') as f:
        json.dump(json_cluster_data, f, indent=4)




    # inference code
    
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
    # print(transformed_array_p)
    df_transformed_p = pd.DataFrame(transformed_array_p, columns=df_to_transform_p.columns, index=df_predict.index)
    # Recombine the DataFrame
    df_result_p = pd.concat([df_transformed_p, df_excluded_p], axis=1)
    
    X_p=df_result_p.values
    X_p=X_p*weights
    
    predicted_label=kmeans.predict(X_p)
    
    state_full_data=fulldata[state]
    
    def filter_by_labels(data, labels):
        # Use list comprehension to get dictionaries with matching labels
        filtered_data = [entry for entry in data if entry['labels'] == labels]
        return filtered_data

    state_filtered_data = filter_by_labels(state_full_data, predicted_label[0])
   
    zpid_list = []
    for i in range(len(state_filtered_data)):
        zpid_list.append(state_filtered_data[i]['zpid'])

    with open("./data_without_ic.json", "r") as f:  ## change embeddings file here
        data_embed = json.load(f)
    # new_data_point = file_embeddings[str(zpid_list[0])]
    # print(zpid_list)
    dict_embeddings = {}
    for i in zpid_list:
        if str(i) in data_embed.keys():
            dict_embeddings[i] = data_embed[str(i)]

    payload = {
        'embeddings': dict_embeddings,
        'message': description
    }

    # Send a POST request
    response = requests.post(
        server_url,
        json=payload,
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code == 200:
        print("Data sent successfully!")
    else:
        print("Failed to send data. Status code:", response.status_code)

    zpid_sim = response.json()['similarities']
    json_file_path = "./combined_summary_data.json"

    zpid_list = [str(zpid) for zpid, _ in zpid_sim]
    with open(json_file_path, "r",encoding="utf-8", errors='ignore') as file:
        data = json.load(file)


    filtered_houses = [data[zpid] for zpid in zpid_list if zpid in data]

    recommended_properties = filtered_houses
    recommendations = pd.DataFrame(recommended_properties, index=zpid_list)



    # score calculation
    top_10_recommendations = recommendations.head(10)
    
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

results.to_csv("Results_Without_IC.csv", index=False)
print("Done")