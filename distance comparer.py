import pandas as pd
from geopy.distance import geodesic

df_leisure = pd.read_csv('combined_leisure.csv')
df_shops = pd.read_csv('combined_shops.csv')
df_schools = pd.read_csv('combined_schools.csv')
df_transit = pd.read_csv('combined_transit.csv')

df_listings = pd.read_csv('ListingInCities.csv')

# Define a function to calculate the number of facilities within a certain range
def count_facilities_within_range(listing, df_facilities, range_km):
    count = 0
    for index, facility in df_facilities.iterrows():
        facility_coords = (facility['latitude'], facility['longitude'])
        listing_coords = (listing['latitude'], listing['longitude'])
        distance = geodesic(listing_coords, facility_coords).kilometers
        if distance <= range_km and facility['city'] == listing['addressCity']:
            count += 1
    return count

# Define the range in kilometers
range_km = 10

# Iterate through each listing and calculate the number of each type of facility within the range
for index, listing in df_listings.iterrows():
    listing['leisure_within_10km'] = count_facilities_within_range(listing, df_leisure, range_km)
    listing['shops_within_10km'] = count_facilities_within_range(listing, df_shops, range_km)
    listing['schools_within_10km'] = count_facilities_within_range(listing, df_schools, range_km)
    listing['transit_within_10km'] = count_facilities_within_range(listing, df_transit, range_km)

# Save the updated DataFrame with new columns
df_listings.to_csv('Updated_ListingInCities.csv', index=False)
