from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="kaggle_learn")
location = geolocator.geocode("Pyramid of Khufu")
location = geolocator.geocode("Praha")

print(location.point)
print(location.address)

print("Latitude:", location.point.latitude)
print("Longitude:", location.point.longitude)