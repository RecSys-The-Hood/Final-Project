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
df_listings['addressCity'] = df_listings['address'].apply(lambda x: ''.join(x['city'].split()).capitalize())

# Define a function to calculate the number of facilities within a certain range
def count_facilities_within_range(listing, df_facilities, range_km):
    count = 0

    facility_coords_list = [(facility['latitude'], facility['longitude']) for index, facility in df_facilities.iterrows()]
    facility_coords_in_city_list = [facility_coords_list[i] for i in range(len(facility_coords_list)) if df_facilities['city'][i] == listing['addressCity']]
    listing_coords = (listing['latitude'], listing['longitude'])
    distances_list = [geodesic(listing_coords, facility_coords).kilometers for facility_coords in facility_coords_in_city_list]

    count = sum([1 for distance in distances_list if distance <= range_km])

    # for index, facility in df_facilities.iterrows():
    #     facility_coords = (facility['latitude'], facility['longitude'])
    #     listing_coords = (listing['latitude'], listing['longitude'])
    #     distance = geodesic(listing_coords, facility_coords).kilometers
    #     if distance <= range_km and facility['city'] == listing['addressCity']:
    #         count += 1
    # print("done")
    return count

df_listings['leisure_within_5km'] = pd.Series(dtype='float64')
df_listings['shops_within_5km'] = pd.Series(dtype='float64')
df_listings['schools_within_5km'] = pd.Series(dtype='float64')
df_listings['transit_within_2km'] = pd.Series(dtype='float64')

# Iterate through each listing and extract address fields and calculate facility counts
# for index, listing in df_listings.iterrows():
#     for key, value in listing['address'].items():
#         new_key = 'address' + key.capitalize()
#         listing[new_key] = value
#     if(index==5):
#         break
#     print(str(index)+'/'+str(len(df_listings)))

#     df_listings['leisure_within_5km'] = df_leisure.apply(lambda x: count_facilities_within_range(x, df_leisure, 5))

#     df_listings.at[index,'leisure_within_5km'] = count_facilities_within_range(listing, df_leisure, 5)
#     df_listings.at[index,'shops_within_5km'] = count_facilities_within_range(listing, df_shops, 5)
#     df_listings.at[index,'schools_within_5km'] = count_facilities_within_range(listing, df_schools, 5)
#     df_listings.at[index,'transit_within_2km'] = count_facilities_within_range(listing, df_transit, 2)

df_listings['leisure_within_5km'] = df_listings.apply(lambda x: count_facilities_within_range(x, df_leisure, 5), axis=1)
df_listings['shops_within_5km'] = df_listings.apply(lambda x: count_facilities_within_range(x, df_shops, 5), axis=1)
df_listings['schools_within_5km'] = df_listings.apply(lambda x: count_facilities_within_range(x, df_schools, 5), axis=1)
df_listings['transit_within_2km'] = df_listings.apply(lambda x: count_facilities_within_range(x, df_transit, 2), axis=1)

# Drop the original 'address' column
df_listings.drop('address', axis=1, inplace=True)
print(df_listings.head(10))
print("Writing")
# Save the updated DataFrame with new columns
df_listings.to_csv('data_new2.csv', index=False)
