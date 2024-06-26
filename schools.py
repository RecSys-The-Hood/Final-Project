import os
import json
import pandas as pd

def filter_geojson_by_coordinates(geojson_file, city_name, city_coordinate_ranges):
    # Load the GeoJSON data
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)
    
    filtered_features = []
    
    # Extract city coordinates
    city_coordinates = city_coordinate_ranges.get(city_name.lower())
    if city_coordinates is None:
        print(f"Coordinates not found for {city_name}. Skipping file: {geojson_file}")
        return None
    
    # Iterate over each feature in the GeoJSON data
    for feature in geojson_data['features']:
        # Extract latitude and longitude from the feature
        longitude, latitude = feature['geometry']['coordinates']
        
        # Check if the coordinates fall within the city's range
        lat_range = city_coordinates['latitude']
        lon_range = city_coordinates['longitude']
        if (lat_range[0] <= latitude <= lat_range[1]) and (lon_range[0] <= longitude <= lon_range[1]):
            # Extract required properties
            properties = feature.get('properties', {})
            name = properties.get('name')
            gnis_feature_id = properties.get('gnis:feature_id')
            gnis_county_id = properties.get('gnis:county_id')
            gnis_state_id = properties.get('gnis:state_id')
            created = properties.get('gnis:created')
            
            # Create data dictionary
            data = {
                'latitude': latitude,
                'longitude': longitude,
                'city': city_name,
                'name': name,
                'feature_id': gnis_feature_id,
                'county_id': gnis_county_id,
                'state_id': gnis_state_id,
                'created': created
            }
            
            # Append to filtered_features list
            filtered_features.append(data)
        
    return filtered_features

def process_schools_folder(folder_path, city_coordinate_ranges):
    all_data = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.geojson'):
            city_name = filename.split('_')[0].capitalize()  # Extract city name from file name
            file_path = os.path.join(folder_path, filename)
            
            # Filter GeoJSON data by coordinates for the city
            filtered_features = filter_geojson_by_coordinates(file_path, city_name, city_coordinate_ranges)
            if filtered_features:
                all_data.extend(filtered_features)
    
    return all_data

# Define city coordinate ranges
city_coordinate_ranges = {
    "atlanta": {"latitude": (33, 34), "longitude": (-85, -84)},
    "boston": {"latitude": (42, 43), "longitude": (-71, -70)},
    "chicago": {"latitude": (41, 42), "longitude": (-88, -87)},
    "dallas": {"latitude": (32, 33), "longitude": (-97, -96)},
    "detroit": {"latitude": (42, 43), "longitude": (-84, -82)},
    "houston": {"latitude": (29, 30), "longitude": (-96, -94)},
    "lasvegas": {"latitude": (36, 37), "longitude": (-116, -114)},
    "losangeles": {"latitude": (33, 34), "longitude": (-119, -117)},
    "miami": {"latitude": (25, 26), "longitude": (-81, -79)},
    "newyork": {"latitude": (40, 41), "longitude": (-75, -73)},
    "philadelphia": {"latitude": (39, 40), "longitude": (-76, -74)},
    "phoenix": {"latitude": (33, 34), "longitude": (-113, -111)},
    "sanfrancisco": {"latitude": (37, 38), "longitude": (-123, -121)},
    "sanjose": {"latitude": (37, 38), "longitude": (-122, -120)},
    "seattle": {"latitude": (47, 48), "longitude": (-123, -121)},
    "washington": {"latitude": (38, 39), "longitude": (-78, -76)}
}

# Folder path containing Schools GeoJSON files
schools_folder_path = './Schools/'

# Process Schools GeoJSON files in the folder
schools_data = process_schools_folder(schools_folder_path, city_coordinate_ranges)

# Convert to DataFrame
schools_df = pd.DataFrame(schools_data)

# Save DataFrame to CSV
schools_df.to_csv('combined_schools.csv', index=False)
