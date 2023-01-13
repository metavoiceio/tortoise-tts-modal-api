import os
import re

import fastapi
import fastapi.middleware.cors
import modal
import torch
import torchaudio
from fastapi import Request
from fastapi.responses import Response

from model import ParallelTortoise, stub

###### MODAL CPU API Split Jobs logic.


@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    ),
    timeout=15 * 60,
)
def long_synthesis(text, voices):
    """
    Modal function for running Tortoise for long-form content.
    """
    import asyncio
    import time

    import torch

    if (type(text) != list) and (len(text) <= 1):
        raise Exception(
            "Long synthesis requires a list of text with at least 2 elements."
        )

    all_parts = []
    inputs = [(t, voices) for t in text]
    wavs = ParallelTortoise().run_tts.starmap(inputs)
    all_parts.extend(list(wavs))
    full_audio = torch.cat(all_parts, dim=-1)

    return full_audio


if __name__ == "__main__":
    with stub.run():
        text = ["Test.", "Test."]

        voices = ["tom", "emma", "scarlett"]

        inputs = [(text, v) for v in voices]
        auds = long_synthesis.starmap(inputs)

        for voice, aud in zip(voices, auds):
            torchaudio.save(os.path.join(f"output_{voice}.wav"), aud, 24000)
