import streamlit as st
import pandas as pd

# Initialize session state for navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

# Read CSV data from a predefined location
# Make sure to replace 'your_file_path.csv' with the actual path to your CSV file
csv_file_path = "Final_Combined_Dataset.csv"  # Example: "data/properties.csv"
df = pd.read_csv(csv_file_path)

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
        }
        st.session_state["current_page"] = "recommendations"

# Display property recommendations if redirected to the recommendations page
elif st.session_state["current_page"] == "recommendations":
    st.title("Recommended Properties")

    # Get form data from session state
    form_data = st.session_state["form_data"]

    # Recommendation logic: filter by preferred property type and price range
    recommended_properties = df[
        (df["homeType"] == form_data["property_type"])
        & (df["price"] <= form_data["income"])
    ]

    # Display recommended properties with images and details
    for _, property in recommended_properties.iterrows():
        with st.container():
            # Organize content with columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Display the property image
                st.image(property["hdpUrl"], use_column_width=True)

            with col2:
                # Display property details
                st.header(property["Name"])
                st.write(f"Location: {property['address.city']}, {property['address.state']}")
                st.write(f"Price: ${property['price']}")
                st.write(f"Bedrooms: {property['bedrooms']}, Bathrooms: {property['bathrooms']}")
                st.write(f"Size: {property['livingArea']} sqft")
                st.write(property["description"])

                # Add an interactive element for contacting about the property
                if st.button(f"Contact about {property['Name']}"):
                    st.write("Contact form coming soon!")

    # Button to go back and modify the search
    if st.button("Modify Search"):
        st.session_state["current_page"] = "home"
