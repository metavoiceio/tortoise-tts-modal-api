import json
import os
from typing import Union

import numpy as np
import requests
from pydub import AudioSegment


class Tortoise:
    def __init__(self, api_key: str) -> None:
        self.url = "https://vatsalaggarwal--tts-app.modal.run"
        self.headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=600, max=100",
        }
        self.body = {
            "api_key": api_key,
        }

    def run_tts(
        self, text: str, voices: str, target_file: Union[str, None] = None
    ) -> np.ndarray:
        """
        Returns a 24khz 32-bit PCM WAV file.
        """
        self.body["text"] = text
        self.body["voices"] = voices
        self.body["target_file"] = target_file

        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(self.body)
        )

        return np.frombuffer(response.content, dtype=np.int32)


if __name__ == "__main__":
    t = Tortoise(os.environ["METAVOICE_API_KEY"])

    # Use an existing voice
    aud = t.run_tts("Hello.", "random")
    print(aud.shape)

    # Or use a custom voice -- zero-shot (Mark Zuckerberg)
    aud = t.run_tts(
        "Hello.", "", "https://mv-public2.s3.eu-west-1.amazonaws.com/mark.mp3"
    )
    print(aud.shape)
