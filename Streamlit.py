import streamlit as st
import pandas as pd

# Set the path to your CSV file
CSV_FILE_PATH = "combined_summary_data.csv"  # Ensure the file exists at this path

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

st.session_state["uploaded_data"] = load_data(CSV_FILE_PATH)

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
    if st.session_state["uploaded_data"] is None:
        st.write("Failed to load the property data.")
        st.stop()

    dummy_df = st.session_state["uploaded_data"]

    # List of states for the drop-down
    states = ['AL', 'DC', 'IL', 'NV', 'AZ', 'PA', 'GA', 'CA', 'TX', 'NY', 'FL', 'MA', 'MI']

    # Form to collect user input for property recommendations
    with st.form("user_info_form"):
        state_selected = st.selectbox("Select State", states, index=0)  # Default to first state
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
        description = st.text_area("Describe the house you want", "")

        submit = st.form_submit_button("Get Recommendations")

    # If the form is submitted, store form data in session state and navigate to recommendations
    if submit:
        form_data = {
            "state_selected": state_selected,
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
            "description": description,
        }
        st.session_state["form_data"] = form_data
        st.session_state["current_page"] = "recommendations"

elif st.session_state["current_page"] == "recommendations":
    st.title("Recommended Properties")

    # Get form data from session state
    form_data = st.session_state["form_data"]

    dummy_df = st.session_state["uploaded_data"]

    # Recommendation logic using the selected state
    recommended_properties = dummy_df[
        dummy_df["address.state"] == form_data["state_selected"]
    ]

    # Display recommended properties with images and details
    if recommended_properties.empty:
        st.write("No properties found in the selected state. Try modifying your search.")
        st.session_state["current_page"] = "home"
        st.stop()

    for idx, property in recommended_properties.iterrows():
        with st.container():
            st.header(property["Name"])
            st.write(f"Location: {property['address.city']}, {property['address.state']}")
            st.write(f"Price: ${property['price']}")
            st.write(f"Bedrooms: {property['bedrooms']}, {property['bathrooms']}")
            st.write(f"Size: {property['livingArea']} sqft")
            st.write(property["description"])

            if "image_urls" in property and len(property["image_urls"]) > 0:
                st.image(property["image_urls"][0], use_column_width=True)

                if st.button("View All Images"):
                    st.write("## All Images")
                    for image_url in property["image_urls"]:
                        st.image(image_url, use_column_width=True)
            else:
                st.write("No images available.")

    # Navigation buttons
    col1, col2 = st.columns(2)
    if col1.button("Modify Search"):
        st.session_state["current_page"] = "home"
    if col2.button("Back to Form"):
        st.session_state["current_page"] = "home"
