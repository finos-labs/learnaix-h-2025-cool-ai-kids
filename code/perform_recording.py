################### Implementation using speech_recognition, an offline library ###################

import threading
import speech_recognition as sr
import logging

# ------------------------
# Setup logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ------------------------
# Global recognizer and mic
# ------------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()

# ------------------------
# Globals to store recording state
# ------------------------
audio_data = None
recording_thread = None
recording_lock = threading.Lock()


def speech_recognition_record_audio():
    """
    Start a new recording in a background thread.
    Can be called multiple times safely.
    """
    global audio_data, recording_thread

    logging.info("Starting new recording...")
    with recording_lock:
        audio_data = None  # reset previous recording

    def _record():
        global audio_data
        try:
            with mic as source:
                logging.info("Calibrating for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)

                recognizer.pause_threshold = 2.0
                recognizer.non_speaking_duration = 0.5
                recognizer.energy_threshold = 300

                # Warm-up listen to avoid missing first word
                logging.info("Warm-up listen...")
                try:
                    recognizer.listen(source, timeout=0.2, phrase_time_limit=0.2)
                except sr.WaitTimeoutError:
                    pass

                logging.info("🎤 Recording started...")
                audio_data = recognizer.listen(source, phrase_time_limit=None)
                logging.info("✅ Recording complete.")

        except Exception as e:
            logging.error(f"Error during recording: {e}")

    recording_thread = threading.Thread(target=_record, daemon=True)
    recording_thread.start()
    return "🎤 Listening your query!!"


def speech_recognition_transcribe_audio():
    """
    Transcribes audio after recording finishes.
    Can be called multiple times after each recording.
    """
    global audio_data, recording_thread

    logging.info("Transcription requested...")

    if recording_thread and recording_thread.is_alive():
        logging.info("Waiting for recording thread to finish...")
        recording_thread.join()

    if not audio_data:
        logging.warning("No audio captured to transcribe.")
        return "⚠️ Recording not finished yet or no speech detected.", ""

    try:
        logging.info("Transcribing audio...")
        text = recognizer.recognize_google(audio_data)
        return "📝 Transcription complete!!", text
    except sr.UnknownValueError:
        logging.warning("Could not understand audio.")
        return "❌ Could not understand audio.", ""
    except sr.RequestError as e:
        logging.error(f"API error during transcription: {e}")
        return f"❌ API error: {e}", ""

