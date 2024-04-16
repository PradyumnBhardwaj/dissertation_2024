import requests
from datetime import datetime

# Replace 'YOUR_API_KEY' with your actual OpenWeatherMap API key
API_KEY = 'YOUR API KEY'
FORECAST_BASE_URL = 'http://api.openweathermap.org/data/2.5/forecast?'
CURRENT_WEATHER_BASE_URL = 'http://api.openweathermap.org/data/2.5/weather?'
GEOCODING_BASE_URL = 'http://api.openweathermap.org/geo/1.0/direct?'

def get_coordinates(city_name):
    complete_url = f"{GEOCODING_BASE_URL}q={city_name}&limit=1&appid={API_KEY}"
    response = requests.get(complete_url)
    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
        else:
            print("City not found.")
            return None, None
    else:
        print(f"Error fetching coordinates: {response.status_code}, {response.text}")
        return None, None

def get_current_weather(latitude, longitude):
    complete_url = f"{CURRENT_WEATHER_BASE_URL}lat={latitude}&lon={longitude}&appid={API_KEY}&units=metric"
    response = requests.get(complete_url)
    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        description = data['weather'][0]['description']
        return temperature, description
    else:
        print(f"Error fetching current weather: {response.status_code}, {response.text}")
        return None

def get_forecast(latitude, longitude):
    complete_url = f"{FORECAST_BASE_URL}lat={latitude}&lon={longitude}&appid={API_KEY}&units=metric"
    response = requests.get(complete_url)
    if response.status_code == 200:
        data = response.json()
        first_forecast = data['list'][0]
        temperature = first_forecast['main']['temp']
        description = first_forecast['weather'][0]['description']
        forecast_time = datetime.fromtimestamp(first_forecast['dt']).strftime('%Y-%m-%d %H:%M:%S')
        return forecast_time, temperature, description
    else:
        print(f"Error fetching forecast: {response.status_code}, {response.text}")
        return None

if __name__ == "__main__":
    city_name = input("Enter city name: ")
    latitude, longitude = get_coordinates(city_name)
    if latitude is not None and longitude is not None:
        current_weather = get_current_weather(latitude, longitude)
        if current_weather:
            current_temperature, current_description = current_weather
            print(f"Current Weather: Temperature = {current_temperature}°C, Description = {current_description}")
        else:
            print("Current weather information could not be retrieved.")

        forecast_info = get_forecast(latitude, longitude)
        if forecast_info:
            forecast_time, forecast_temperature, forecast_description = forecast_info
            print(f"Upcoming Forecast for: {forecast_time}")
            print(f"Weather: Temperature = {forecast_temperature}°C, Description = {forecast_description}")
        else:
            print("Forecast information could not be retrieved.")
    else:
        print("Could not find the specified city.")
