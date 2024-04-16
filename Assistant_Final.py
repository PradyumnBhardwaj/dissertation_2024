import os
import speech_recognition as sr
import webbrowser  
import ollama
import time
from youtube_search import search_youtube  
from Weather import get_current_weather, get_coordinates
from send_email import send_email
# Initialize recognizer
recognizer = sr.Recognizer()

def recognize_speech_and_respond(recognizer, source):
    recognizer.adjust_for_ambient_noise(source)
    print("Say something or 'stop' to end...")
    os.system("Say Ask something or 'stop' to end...")
    try:
        # Listening...
        audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        text = recognizer.recognize_google(audio_data).lower()
        print(f"You said: {text}")

        if "stop" in text:
            os.system("say 'Stopping, goodbye!'")
            print("say 'Stopping, goodbye!'")
            return "stop"
        elif "search youtube" in text:
            # Extract the search query from the spoken text
            os.system("say 'What do you want me to search?'")
            audio_data_youtube = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text_youtube = recognizer.recognize_google(audio_data_youtube).lower()
            search_youtube(text_youtube)
            time.sleep(15)
            return "searched youtube"
        
        elif "weather in" in text:
            
            city_name = text.replace("what's the weather in", "").strip()
            if city_name:
                latitude, longitude = get_coordinates(city_name)
                if latitude is not None and longitude is not None:
                    current_weather = get_current_weather(latitude, longitude)
                    if current_weather:
                        current_temperature, current_description = current_weather
                        weather_response = f"Current Weather in {city_name}: Temperature = {current_temperature}°C, Description = {current_description}"
                        response_new = ollama.chat(model='falcon:7b', messages=[
                            {'role': 'user', 'content': weather_response},
                        ])
            # time.sleep(5)
                        response_text = response_new['message']['content']
                        print(response_text)
                        print(weather_response)
            # Speak out the response from Ollama
                        os.system(f"say '{response_text}'")
                        
                    else:
                        os.system("say 'Current weather information could not be retrieved.'")
                else:
                    os.system("say 'Could not find the specified city.'")
            else:
                os.system("say 'Please specify a city name after the weather command.'")
            return "weather checked"
        
        elif "send an email" in text:
            # Extract the search query from the spoken text
            #city_name = text.replace("send an email", "").strip()
            os.system("say 'Who do you want to send?'")
            # audio_data_receiver = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            # text_receiver= recognizer.recognize_google(audio_data_receiver).lower()
            os.system("say 'What should I write'")
            audio_data_mail = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text_mail= recognizer.recognize_google(audio_data_mail).lower()           
            response_mail = ollama.chat(model='falcon:7b', messages=[
                {'role': 'user', 'content': 'write an email on the topic' + text_mail},
            ])
            # time.sleep(5)
            response_text_mail = response_mail['message']['content'] 
            
            subject= extract_subject(response_text_mail)  
            print(subject) 
            body= extract_message(response_text_mail)   
            print(body)    
            receiver='vishubhardwaj229@gmail.com'
            send_email(receiver, subject, body)

            return "sent email"
        else:
            # Use Ollama model to get a response
            response = ollama.chat(model='falcon:7b', messages=[
                {'role': 'user', 'content': text},
            ])
            # time.sleep(5)
            response_text = response['message']['content']
            print(response_text)
            # Speak out the response from Ollama
            os.system(f"say '{response_text}'")
            return text
    except sr.WaitTimeoutError:
        os.system("say 'I did not catch that, please try again.'")
        return ""
    except sr.UnknownValueError:
        os.system("Say 'Sorry, I could not understand that.'")
        return ""
    except sr.RequestError as e:
        os.system(f"say 'I could not request results from the speech recognition service; {e}'")
        return ""

def extract_subject(email_text: str) -> str:
    """
    Extracts the subject line from an email text.

    Parameters:
    - email_text (str): The text of the email.

    Returns:
    - str: The subject of the email.
    """
    lines = email_text.split("\n")
    for line in lines:
        if line.startswith("Subject:"):
            return line.split("Subject: ")[1].strip()
    return ""

def extract_message(email_text: str) -> str:
    """
    Extracts the message body from an email text, excluding the subject line.

    Parameters:
    - email_text (str): The text of the email.

    Returns:
    - str: The message body of the email.
    """
    lines = email_text.split("\n")
    message_body = ""
    capture_message = False
    for line in lines:
        if capture_message:
            message_body += line + "\n"
        elif line.startswith("Subject:"):
            capture_message = True
    return message_body.strip()


# Main loop
with sr.Microphone() as source:
    while True:
        text = recognize_speech_and_respond(recognizer, source)
        if text == "stop":
            break
