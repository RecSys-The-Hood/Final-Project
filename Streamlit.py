# from math import cos
import random
# from IPython import embed
import streamlit as st
# from sentence_transformers import SentenceTransformer
import pandas as pd
# import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# from sympy import asec
# import torch
# from sentence_transformers.util import cos_sim
import json
import joblib
import requests
from zmq import TYPE 
# import csv
import ast
def filter_by_labels(data, labels):
        # Use list comprehension to get dictionaries with matching labels
        filtered_data = [entry for entry in data if entry['labels'] == labels]
        return filtered_data

label_encoder = LabelEncoder()
# Set the path to your CSV file
# CSV_FILE_PATH = "combined_summary_data.csv"  # Ensure the file exists at this path
# model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# Initialize session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"
    st.session_state["form_data"] = {}
    st.session_state["uploaded_data"] = None

# Load the CSV data from the predefined path and cache it
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure the file path is correct.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()



# Sidebar for navigation and information
st.sidebar.title("Navigation")
st.sidebar.markdown("## About This App")
st.sidebar.write(
    "This app helps you find real estate property recommendations based on "
    "your preferences and location. Use the form to input your details, "
    "and get personalized property recommendations."
)

# Sidebar instructions and tips
st.sidebar.markdown("## How to Use")
st.sidebar.write(
    "- Fill out the form with your information and preferences.\n"
    "- After submission, the app will provide property recommendations.\n"
    "- Use the buttons to navigate or modify your search."
)

st.sidebar.markdown("## Notes")
st.sidebar.write(
    "Ensure that the provided data includes relevant property details such as state, "
    "city, price, bedrooms, bathrooms, and image URLs.\n"
    "If the data does not include these details, recommendations may not be accurate."
)

# Navigation logic based on session state
if st.session_state["current_page"] == "home":
    st.title("Real Estate Property Recommendations")

    # Check if data is loaded properly
    # if st.session_state["uploaded_data"] is None:
    #     st.write("Failed to load the property data.")
    #     st.stop()

    # dummy_df = st.session_state["uploaded_data"]

    # List of states for the drop-down
    states = ['AL', 'DC', 'IL', 'NV', 'AZ', 'PA', 'GA', 'CA', 'TX', 'NY', 'FL', 'MA', 'MI']

    # Form to collect user input for property recommendations
    with st.form("user_info_form"):
        state_selected = st.selectbox("Select State", states, index=0)  # Default to first state
        bedrooms = st.number_input("Bedrooms", min_value=1)
        bathrooms = st.number_input("Bathrooms", min_value=1)
        budget = st.number_input("Budget ($)", min_value=40000, step=1000)
        # name = st.text_input("Name")
        property_type = st.selectbox("Preferred Property Type", ['APARTMENT' ,'CONDO' ,'MANUFACTURED', 'MULTI_FAMILY' ,'SINGLE_FAMILY',
 'TOWNHOUSE'])
        living_area = st.number_input("Living Area (sqft)", min_value=100)
        min_proximity = 0  # Minimum distance
        max_proximity = 100  # Maximum distance
        step = 1  # Step size for the slider
        number_recreation = st.slider(
            "Number of Recreational Areas",
            min_proximity,
            max_proximity,
            step=step,
            value=5,  # Default value
        )
        number_shops = st.slider(
            "Number of Shops",
            min_proximity,
            max_proximity,
            step=step,
            value=5,  # Default value
        )
        number_schools = st.slider(
            "Number of Schools",
            min_proximity,
            max_proximity,
            step=step,
            value=5,  # Default value
        )

        vicinity_transit = st.slider(
            "Transit",
            min_proximity,
            max_proximity,
            step=step,
            value=5,  # Default value
        )

        description = st.text_area("Describe the house you want", "")

        submit = st.form_submit_button("Get Recommendations")

    # If the form is submitted, store form data in session state and navigate to recommendations
    if submit:
        form_data = {
            "state_selected": state_selected,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms, 
            "price": budget,
            "homeType": property_type,
            "livingArea": living_area,
            "description": description,
            "leisure_within_5km": number_recreation*10,
            "shops_within_5km": number_shops*10,
            "schools_within_5km": number_schools*10,
            "transit_within_2km":vicinity_transit*10
        }
        st.session_state["form_data"] = form_data
        st.session_state["current_page"] = "recommendations"

elif st.session_state["current_page"] == "recommendations":
    st.title("Recommended Properties")

    # Get form data from session state
    form_data = st.session_state["form_data"]

    description = form_data["description"]
    
    df_data=pd.DataFrame([form_data])
    df_data=df_data.set_index('state_selected')
    df_data = df_data.drop(columns=['description'])

    homeType_Encoding= {
        'APARTMENT': 0,
        'CONDO': 1,
        'MANUFACTURED': 2,
        'MULTI_FAMILY': 3,
        'SINGLE_FAMILY': 4,
        'TOWNHOUSE': 5
    }

    df_data['homeType'] = df_data['homeType'].map(homeType_Encoding)

    json_file_path = "./Data/ClusterPoints_Dataset.json"

    with open(json_file_path, "r") as file:
        kmeansdata = json.load(file)

    json_file_path_1 = "./Data/Labelled_Dataset_with_zpid.json"

    with open(json_file_path_1, "r") as file:
        fulldata = json.load(file)

    # cluster_points= kmeansdata[form_data["state_selected"]]
    # print(cluster_points)
    scaler=StandardScaler()
    X=scaler.fit_transform(df_data)

    kmeans = joblib.load(f'./Data/{form_data['state_selected']}_kmeans.pkl')
    
    predicted_label=kmeans.predict(X)
    print(predicted_label)
    state_full_data=fulldata[form_data["state_selected"]]
    # print(state_full_data[0])

    state_filtered_data = filter_by_labels(state_full_data, predicted_label[0])
    print(state_filtered_data[0])
    zpid_list = []
    for i in range(len(state_filtered_data)):
        zpid_list.append(state_filtered_data[i]['zpid'])
    
    with open("./Data/data_embed.json", "r") as f:
        data_embed = json.load(f)
    # new_data_point = file_embeddings[str(zpid_list[0])]
    print(zpid_list)
    dict_embeddings = {}
    for i in zpid_list:
        if str(i) in data_embed.keys():
            dict_embeddings[i] = data_embed[str(i)]
    
    dict_embeddings_values = {}
    a = dict_embeddings.values()
    # l1= [float(i) for i in a]
    print("Tsting")
    # print(dict_embeddings)
    payload = {
        'embeddings': dict_embeddings,
        'message': description
    }

    # Define the URL of the backend server
    server_url = "http://127.0.0.1:5000/predict"  # Replace with your server URL

    # Send a POST request
    response = requests.post(
        server_url,
        json=payload,  # Sending data as JSON
        headers={'Content-Type': 'application/json'}  # Specifying content type
    )

    # Check if the request was successful
    if response.status_code == 200:
        print("Data sent successfully!")
        # Do something with the response
        # print("Server response:", response.json())
    else:
        print("Failed to send data. Status code:", response.status_code)
    # embedding_description = model.encode(form_data["description"])
    # similarity_values = cos_sim(embedding_description[0],torch.tensor(l1))
    # print(new_data_point)
    # print(zpid_list)
    # for i in zpid_list:
    #     dict_embeddings_values[i] = similarity_values[i]
    
    zpid_sim = response.json()['similarities']
    # print(type(zpid_sim))
    json_file_path = "./Data/combined_summary_data.json"
    
    zpid_list = [str(zpid) for zpid, _ in zpid_sim]
    with open(json_file_path, "r",encoding="utf-8", errors='ignore') as file:
        data = json.load(file)

    filtered_houses = [house for house in data if str(house.get("zpid")) in zpid_list]
    
    st.session_state["uploaded_data"] = pd.DataFrame(filtered_houses)

    dummy_df = st.session_state["uploaded_data"]

    # Recommendation logic using the selected state
    recommended_properties = dummy_df

    # Display recommended properties with images and details
    if recommended_properties.empty:
        st.write("No properties found in the selected state. Try modifying your search.")
        st.session_state["current_page"] = "home"
        st.stop()

    col1, col2 = st.columns(2)
    if col1.button("Modify Search"):
        st.session_state["current_page"] = "home"
    if col2.button("Back to Form"):
        st.session_state["current_page"] = "home"
        
    for idx, property in recommended_properties.iterrows():
        with st.container():
            st.header(property["address.streetAddress"])
            st.write(f"Location:- City: {property['address.city']}, State: {property['address.state']}")
            st.write(f"Price: ${property['price']}")
            st.write(f"Bedrooms: {property['bedrooms']} , Bathrooms: {property['bathrooms']}")
            st.write(f"Size: {property['livingArea']} sqft")
            st.write(f"Property Type: {property['homeType']}")
            st.write(property["summary"])

            if "originalPhotos" in property and len(property["originalPhotos"]) > 0:
                urls=ast.literal_eval(property["originalPhotos"])
                st.image(urls[0], use_column_width=True)

                if st.button("View All Images", key=idx):
                    st.write("## All Images")
                    for image_url in urls:
                        st.image(image_url, use_column_width=True)
            else:
                st.write("No images available.")

    # Navigation buttons
    col1, col2 = st.columns(2)
    if col1.button("Modify Search", key=10000):
        st.session_state["current_page"] = "home"
    if col2.button("Back to Form",key=20000):
        st.session_state["current_page"] = "home"
