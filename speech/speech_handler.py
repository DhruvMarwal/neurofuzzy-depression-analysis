"""
SECTION 6: SPEECH HANDLER (Microphone Input)
=============================================
Two modes:
  1. Browser-side (recommended) — uses Web Speech API via JavaScript
     in the frontend. This Python file handles the fallback server-side
     transcription when the browser API is unavailable.

  2. Server-side — SpeechRecognition library (Google backend or offline
     Vosk for privacy-sensitive deployments).

Usage (server-side standalone):
  python speech_handler.py

Usage (via API):
  POST /speech/transcribe  with audio file upload
"""

import os
import io
import wave
import tempfile
import logging
import time

logger = logging.getLogger(__name__)


class SpeechHandler:
    """
    Handles speech-to-text conversion for the depression analysis chatbot.
    Supports:
      - Microphone capture (server-side, requires PyAudio)
      - Audio file transcription (WAV, MP3, OGG)
      - Real-time streaming (chunked, for live feedback)
    """

    def __init__(
        self,
        language: str = "en-IN",
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
    ):
        """
        Args:
            language:         Speech recognition language code.
                              "en-IN" for Indian English, "en-US" for US English.
            energy_threshold: Mic sensitivity (lower = more sensitive).
            pause_threshold:  Seconds of silence to end utterance.
        """
        try:
            import speech_recognition as sr
            self.sr          = sr
            self.recognizer  = sr.Recognizer()
            self.recognizer.energy_threshold = energy_threshold
            self.recognizer.pause_threshold  = pause_threshold
            self.recognizer.dynamic_energy_threshold = True
            self.language    = language
            self._available  = True
        except ImportError:
            logger.warning("SpeechRecognition not installed. pip install SpeechRecognition pyaudio")
            self._available = False

    # ── Microphone Input ───────────────────────────────────────────────────────

    def listen_from_mic(self, timeout: float = 10.0, phrase_limit: float = 30.0) -> dict:
        """
        Listen from the default microphone and return transcribed text.

        Args:
            timeout:      Max seconds to wait for speech to start.
            phrase_limit: Max seconds per utterance.
        Returns:
            dict with 'text', 'success', 'error'
        """
        if not self._available:
            return {"success": False, "error": "SpeechRecognition not available", "text": ""}

        try:
            with self.sr.Microphone() as source:
                logger.info("Calibrating for ambient noise (1s)...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

                logger.info("Listening... (speak now)")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit,
                )

            text = self._transcribe_audio(audio)
            return {"success": True, "text": text, "error": None}

        except self.sr.WaitTimeoutError:
            return {"success": False, "error": "No speech detected within timeout.", "text": ""}
        except self.sr.UnknownValueError:
            return {"success": False, "error": "Could not understand audio.", "text": ""}
        except Exception as e:
            logger.error(f"Mic error: {e}")
            return {"success": False, "error": str(e), "text": ""}

    # ── File Transcription ─────────────────────────────────────────────────────

    def transcribe_file(self, file_path: str) -> dict:
        """
        Transcribe an audio file.

        Args:
            file_path: Path to .wav, .aiff, or .flac file.
        Returns:
            dict with 'text', 'success', 'error'
        """
        if not self._available:
            return {"success": False, "error": "SpeechRecognition not available", "text": ""}

        try:
            with self.sr.AudioFile(file_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
            text = self._transcribe_audio(audio)
            return {"success": True, "text": text, "error": None}
        except Exception as e:
            return {"success": False, "error": str(e), "text": ""}

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> dict:
        """
        Transcribe raw PCM audio bytes.

        Args:
            audio_bytes: Raw PCM audio data.
            sample_rate: Sample rate in Hz (default 16000).
        Returns:
            dict with 'text', 'success', 'error'
        """
        if not self._available:
            return {"success": False, "error": "SpeechRecognition not available", "text": ""}

        try:
            # Wrap bytes in a WAV container
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                with wave.open(f, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)   # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_bytes)

            result = self.transcribe_file(tmp_path)
            os.remove(tmp_path)
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "text": ""}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _transcribe_audio(self, audio) -> str:
        """Send audio to Google Speech Recognition."""
        return self.recognizer.recognize_google(audio, language=self.language)

    # ── Continuous Listening ───────────────────────────────────────────────────

    def listen_continuous(self, callback, stop_event=None):
        """
        Continuously listen from mic and call callback(text) for each utterance.

        Args:
            callback:   Function called with transcribed text string.
            stop_event: threading.Event; set it to stop listening.
        
        Usage:
            import threading
            stop = threading.Event()
            handler = SpeechHandler()
            t = threading.Thread(target=handler.listen_continuous,
                                 args=(print, stop))
            t.start()
            time.sleep(30)
            stop.set()
        """
        if not self._available:
            logger.error("SpeechRecognition not available.")
            return

        logger.info("Starting continuous listening. Speak naturally...")

        def audio_callback(recognizer, audio):
            try:
                text = recognizer.recognize_google(audio, language=self.language)
                if text.strip():
                    callback(text)
            except self.sr.UnknownValueError:
                pass
            except Exception as e:
                logger.warning(f"Recognition error: {e}")

        with self.sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        stop_listening = self.recognizer.listen_in_background(
            self.sr.Microphone(), audio_callback, phrase_time_limit=30
        )

        try:
            while stop_event is None or not stop_event.is_set():
                time.sleep(0.1)
        finally:
            stop_listening(wait_for_stop=False)
            logger.info("Continuous listening stopped.")


# ── JavaScript snippet for browser-side speech (embedded in frontend) ─────────
BROWSER_SPEECH_JS = """
// Browser-side speech recognition using Web Speech API
// This runs entirely in the browser — no audio is sent to the server.

class BrowserSpeechHandler {
  constructor(onResult, onError) {
    this.onResult  = onResult;
    this.onError   = onError;
    this.isListening = false;

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.error('Web Speech API not supported in this browser.');
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.recognition = new SpeechRecognition();
    this.recognition.lang = 'en-IN';
    this.recognition.continuous = false;
    this.recognition.interimResults = true;
    this.recognition.maxAlternatives = 1;

    this.recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map(r => r[0].transcript)
        .join('');
      const isFinal = event.results[event.results.length - 1].isFinal;
      this.onResult(transcript, isFinal);
    };

    this.recognition.onerror = (event) => {
      this.isListening = false;
      this.onError(event.error);
    };

    this.recognition.onend = () => {
      this.isListening = false;
    };
  }

  start() {
    if (this.recognition && !this.isListening) {
      this.recognition.start();
      this.isListening = true;
    }
  }

  stop() {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
      this.isListening = false;
    }
  }
}
"""


# ── Quick test (server-side mic) ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing SpeechHandler...")
    handler = SpeechHandler(language="en-IN")

    print("\nSay something into your microphone...")
    result = handler.listen_from_mic(timeout=10)

    if result["success"]:
        print(f"\n✓ Transcribed: {result['text']}")
    else:
        print(f"\n✗ Error: {result['error']}")