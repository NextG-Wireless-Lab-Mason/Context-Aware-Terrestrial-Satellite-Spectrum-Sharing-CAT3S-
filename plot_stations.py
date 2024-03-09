import gmplot
import os
import numpy as np
import math


def plot_stations(lat_FSS, lon_FSS, rain, base_stations, inr_each_bs):
    # gmap = gmplot.GoogleMapPlotter.from_geocode("USA", apikey="AIzaSyC4BI4H4SCVgA2nGASWUlGPpJS4f-jPjgw")
    center_lat = np.mean(lat_FSS)
    center_long = np.mean(lon_FSS)
    gmap = gmplot.GoogleMapPlotter(center_lat, center_long, 13, apikey="AIzaSyC4BI4H4SCVgA2nGASWUlGPpJS4f-jPjgw", map_type='satellite')
    if rain:
        weather_url = 'https://friendlystock.com/wp-content/uploads/2021/07/4-weather-emoji-rainy-cartoon-clipart.jpg'
    else:
        weather_url = 'https://friendlystock.com/wp-content/uploads/2021/07/1-weather-emoji-sunny-cartoon-clipart.jpg'

    bounds = {'north': 37.21933173298058, 'south': 37.19933173298058, 'east': -80.50983898577554, 'west': -80.52983898577554}
    gmap.ground_overlay(weather_url, bounds, opacity=0.8)



    buckets = 5
    # https://coolors.co/palette/03071e-370617-6a040f-9d0208-d00000-dc2f02-e85d04-f48c06-faa307-ffba08
    color_palette = ['#FFBA08', '#F48C06', '#DC2F02', '#9D0208', '#370617']
    min_inr_each_bs, max_inr_each_bs = min(inr_each_bs), max(inr_each_bs)
    bucket_size = (max_inr_each_bs - min_inr_each_bs)/buckets

    gmap.marker(lat_FSS, lon_FSS, color='gold', title='FSS Receiver', label='F')

    # Create the map

    # get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # specify the path to save the html file
    save_path = os.path.join(script_dir, "stations_map.html")

    for idx, base in enumerate(base_stations):
        bucket_match = int(math.ceil(inr_each_bs[idx]-min_inr_each_bs)/bucket_size)
        if bucket_match >= buckets:
            bucket_match = buckets-1

        if base['status'] == 1:
            inr = inr_each_bs[idx]
            normalized_inr = round(abs((inr - min_inr_each_bs)/(max_inr_each_bs - min_inr_each_bs)) * 100, 2)

            gmap.marker(
                base['latitude'], base['longitude'], color='green', title=f"BS | INR: {round(inr, 2)}, Status: Active",
                label='B'
            )
            gmap.circle(
                base['latitude'], base['longitude'], radius=normalized_inr * 10,
                color=color_palette[bucket_match]
            )

            # gmap.scatter(
            #     base['latitude'], base['longitude'], color=color_palette[bucket_match % buckets], colorbar=True
            # )
            # gmap.text(37.793575, -122.464334, 'Presidio')
            # gmap.text(37.766942, -122.441472, 'Buena Vista Park', color='blue')
        else:
            gmap.marker(base['latitude'], base['longitude'], color='red', title="Status: Inactive", label='B')
            # gmap.circle(base['latitude'], base['longitude'], radius=inr_each_bs[idx], color='green')

    gmap.draw(save_path)


