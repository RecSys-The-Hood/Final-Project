import streamlit as st
import pandas as pd

# Initialize session state for navigation and form data
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"
    st.session_state["form_data"] = {}

# List of states
states_list = ['AL', 'DC', 'IL', 'NV', 'AZ', 'PA', 'GA', 'CA', 'TX', 'NY', 'FL', 'MA', 'MI']

# Dummy data for demonstration with multiple image URLs
dummy_data = [
    {
        "Name": "House 1", 
        "address.city": "City A", 
        "address.state": "CA", 
        "price": 300000, 
        "bedrooms": 3, 
        "bathrooms": 2, 
        "livingArea": 2000, 
        "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor.", 
        "image_urls": ["https://via.placeholder.com/300", "https://via.placeholder.com/301", "https://via.placeholder.com/302"]
    },
    {
        "Name": "House 2", 
        "address.city": "City B", 
        "address.state": "TX", 
        "price": 250000, 
        "bedrooms": 2, 
        "bathrooms": 1.5, 
        "livingArea": 1500, 
        "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor.", 
        "image_urls": ["https://via.placeholder.com/303", "https://via.placeholder.com/304", "https://via.placeholder.com/305"]
    },
    {
        "Name": "House 3", 
        "address.city": "City C", 
        "address.state": "NY", 
        "price": 400000, 
        "bedrooms": 4, 
        "bathrooms": 3, 
        "livingArea": 2500, 
        "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor.", 
        "image_urls": ["https://via.placeholder.com/306", "https://via.placeholder.com/307", "https://via.placeholder.com/308"]
    },
]

# Convert dummy data to DataFrame
dummy_df = pd.DataFrame(dummy_data)

# Sidebar for additional navigation or information
st.sidebar.title("Navigation")
st.sidebar.markdown("## About This App")
st.sidebar.write(
    """
    This real estate recommendation app helps you find the best properties 
    based on your preferences. Fill out the form to receive personalized 
    recommendations.
    """
)

# Navigation logic based on session state
if st.session_state["current_page"] == "home":
    # Main content area for form and initial information
    st.title("Real Estate Property Recommendations")
    st.write("Please complete the form below to get personalized property recommendations.")

    # Form to collect user input for property recommendations
    with st.form("user_info_form"):
        income = st.number_input("Income ($)", min_value=40000, max_value=200000, step=1000)
        name = st.text_input("Name")
        family_size = st.number_input("Family Size", min_value=1, max_value=6)
        age = st.number_input("Age", min_value=18, max_value=80)
        marital_status = st.selectbox("Marital Status", ["Bachelor", "Married"])
        proximity_hospitals = st.checkbox("Proximity to Hospitals")
        proximity_recreation = st.checkbox("Proximity to Recreational Areas")
        proximity_schools = st.checkbox("Proximity to Schools")
        proximity_workplaces = st.checkbox("Proximity to Workplaces")
        property_type = st.selectbox("Preferred Property Type", ["Apartment", "House", "Condo", "Townhouse"])
        selected_state = st.selectbox("Select State", states_list)
        description = st.text_area("Describe the house you want", "")

        submit = st.form_submit_button("Get Recommendations")

    # If the form is submitted, store form data in session state and navigate to recommendations
    if submit:
        # Store form data in session state
        st.session_state["form_data"] = {
            "income": income,
            "name": name,
            "family_size": family_size,
            "age": age,
            "marital_status": marital_status,
            "proximity_hospitals": proximity_hospitals,
            "proximity_recreation": proximity_recreation,
            "proximity_schools": proximity_schools,
            "proximity_workplaces": proximity_workplaces,
            "property_type": property_type,
            "selected_state": selected_state,
            "description": description
        }
        st.session_state["current_page"] = "recommendations"

# Display property recommendations if redirected to the recommendations page
elif st.session_state["current_page"] == "recommendations":
    st.title("Recommended Properties")

    # Get form data from session state
    form_data = st.session_state["form_data"]

    # Dummy recommendation logic: filter by preferred property type, price range, and selected state
    recommended_properties = dummy_df[
        (dummy_df["address.state"] == form_data["selected_state"])
    ]

    # Display recommended properties with images and details
    for idx, property in recommended_properties.iterrows():
        with st.container():
            # Display property details
            st.header(property["Name"])
            st.write(f"Location: {property['address.city']}, {property['address.state']}")
            st.write(f"Price: ${property['price']}")
            st.write(f"Bedrooms: {property['bedrooms']}, Bathrooms: {property['bathrooms']}")
            st.write(f"Size: {property['livingArea']} sqft")
            st.write(property["description"])

            # Display the first image initially
            st.image(property["image_urls"][0], use_column_width=True)

            # Button to view all images on a new page
            if len(property["image_urls"]) > 1:
                if st.button("View All Images"):
                    # Create a new page to display all images
                    st.write("---")
                    st.write("## All Images")
                    for image_url in property["image_urls"]:
                        st.image(image_url, use_column_width=True)

            # Add an interactive element for contacting about the property
            if st.button(f"Contact about {property['Name']}"):
                st.write("Contact form coming soon!")

    # Buttons to go back and modify the search
    col1, col2 = st.columns(2)
    if col1.button("Modify Search"):
        st.session_state["current_page"] = "home"
    if col2.button("Back to Form"):
        st.session_state["current_page"] = "home"
