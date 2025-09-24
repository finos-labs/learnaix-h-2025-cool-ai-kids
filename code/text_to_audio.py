import pyttsx3

def pyttsx3_text_to_speech(text: str):
    # Initialize TTS engine
    engine = pyttsx3.init()

    # Set properties
    # Get available voices
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    # engine.setProperty("rate", 150)     # Speed (default ~200)
    # engine.setProperty("volume", 1.0)   # Volume (0.0 to 1.0)

    # Speak text
    engine.say(text)
    engine.runAndWait()

# if __name__ == "__main__":
#     text_to_speech_play("Hello! This is Gemini speaking directly, without saving a file.")
