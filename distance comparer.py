import pandas as pd
from geopy.distance import geodesic
import json

# Read CSV files containing facilities data
df_leisure = pd.read_csv('combined_leisure.csv')
df_shops = pd.read_csv('combined_shops.csv')
df_schools = pd.read_csv('combined_schools.csv')
df_transit = pd.read_csv('combined_transit.csv')

# Read JSON file containing listings data
with open('combined_filtered_data.json', 'r') as file:
    listings_data = json.load(file)
df_listings = pd.DataFrame(listings_data)

# Define a function to calculate the number of facilities within a certain range
def count_facilities_within_range(listing, df_facilities, range_km):
    count = 0
    for index, facility in df_facilities.iterrows():
        facility_coords = (facility['latitude'], facility['longitude'])
        listing_coords = (listing['latitude'], listing['longitude'])
        distance = geodesic(listing_coords, facility_coords).kilometers
        if distance <= range_km and facility['city'] == listing['addressCity']:
            count += 1
    print("done")
    return count

df_listings['leisure_within_5km'] = pd.Series(dtype='float64')
df_listings['shops_within_5km'] = pd.Series(dtype='float64')
df_listings['schools_within_5km'] = pd.Series(dtype='float64')
df_listings['transit_within_2km'] = pd.Series(dtype='float64')

# Iterate through each listing and extract address fields and calculate facility counts
for index, listing in df_listings.iterrows():
    listing['leisure_within_5km'] = count_facilities_within_range(listing, df_leisure, 5)
    listing['shops_within_5km'] = count_facilities_within_range(listing, df_shops, 5)
    listing['schools_within_5km'] = count_facilities_within_range(listing, df_schools, 5)
    listing['transit_within_2km'] = count_facilities_within_range(listing, df_transit, 2)

# Save the updated DataFrame with new columns
df_listings.to_csv('data.csv', index=False)
