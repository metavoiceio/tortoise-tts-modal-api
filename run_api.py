import json

import requests


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

    def run_tts(self, text: str, voices: str) -> bytes:
        self.body["text"] = text
        self.body["voices"] = voices
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(self.body)
        )
        return response.content
