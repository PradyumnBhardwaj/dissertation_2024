import os
import speech_recognition as sr
import ollama
import time

# Initialize recognizer
recognizer = sr.Recognizer()

def recognize_speech_and_respond_with_ollama(recognizer, source):
    recognizer.adjust_for_ambient_noise(source)
    print("Say something or 'stop' to end...")
    os.system("Say something or 'stop' to end...")
    try:
        # Listening...
        audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        text = recognizer.recognize_google(audio_data).lower()
        print(f"You said: {text}")

        if "stop" in text:
            os.system("say 'Stopping, goodbye!'")
            print("say 'Stopping, goodbye!'")
            return "stop"
        else:
            # Use Ollama model to get a response
            response = ollama.chat(model='falcon:7b', messages=[
                {'role': 'user', 'content': text},
            ])
            time.sleep(10)
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

# Main loop
with sr.Microphone() as source:
    while True:
        text = recognize_speech_and_respond_with_ollama(recognizer, source)
        if text == "stop":
            break
