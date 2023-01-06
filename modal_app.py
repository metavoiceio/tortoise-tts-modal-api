import io
import tempfile

import modal
from fastapi import Request
from fastapi.responses import Response

stub = modal.Stub("tts")

tortoise_image = (
    modal.Image.conda()
    .apt_install("git")
    .apt_install("libsndfile-dev")
    .apt_install("ffmpeg")
    .run_commands(
        "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116",
        "git clone https://github.com/vatsalaggarwal/tortoise-tts",
        "cd tortoise-tts; pip install -r requirements.txt; pip install -e .",
        # download models as part of the container.
        "python -c 'from tortoise.api import MODELS_DIR, TextToSpeech;tts = TextToSpeech(models_dir=MODELS_DIR); tts.get_random_conditioning_latents()'",
        "pip install pydub",
    )
)

###### MODAL GPU Model.
class Model:
    def __enter__(self):
        from tortoise.api import MODELS_DIR, TextToSpeech
        from tortoise.utils.audio import load_voices

        self.load_voices = load_voices
        self.tts = TextToSpeech(models_dir=MODELS_DIR)
        self.tts.get_random_conditioning_latents()

    def process_synthesis_result(self, result):
        import pydub
        import torchaudio

        with tempfile.NamedTemporaryFile() as converted_wav_tmp:
            torchaudio.save(
                converted_wav_tmp.name + ".wav",
                result,
                24000,
            )
            wav = io.BytesIO()
            _ = pydub.AudioSegment.from_file(
                converted_wav_tmp.name + ".wav", format="wav"
            ).export(wav, format="wav")

        return wav

    @stub.function(image=tortoise_image, gpu="A100")
    def run_tts(self, text, voices):
        CANDIDATES = 1  # NOTE: this code only works for one candidate.
        CVVP_AMOUNT = 0.0
        SEED = None
        PRESET = "fast"

        # TODO: make work for multiple voices
        selected_voices = voices.split(",")

        selected_voice = selected_voices[0]

        if "&" in selected_voice:
            voice_sel = selected_voice.split("&")
        else:
            voice_sel = [selected_voice]
        voice_samples, conditioning_latents = self.load_voices(voice_sel)

        gen, _ = self.tts.tts_with_preset(
            text,
            k=CANDIDATES,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=PRESET,
            use_deterministic_seed=SEED,
            return_deterministic_state=True,
            cvvp_amount=CVVP_AMOUNT,
        )

        wav = self.process_synthesis_result(gen.squeeze(0).cpu())

        return wav


###### MODAL CPU API Key Logic.

supabase_image = modal.Image.debian_slim().pip_install("supabase")


@stub.webhook(
    method="POST",
    image=supabase_image,
    secret=modal.Secret.from_name("supabase-tortoise-secrets"),
)
def app(req: Request):
    import asyncio
    import os
    import time

    from supabase import Client, create_client

    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(url, key)

    body = asyncio.run(req.json())
    text = body["text"]
    voices = body["voices"]
    api_key = body["api_key"]

    # data = supabase.table("users").select("*").eq("api_key", api_key).execute()
    data = supabase.table("users").select("*").execute()
    print(data)
    start = time.time()
    wav = Model().run_tts.call(text, voices)
    end = time.time()
    print(end - start)
    return Response(content=wav.getvalue(), media_type="audio/wav")

    # Query supabase to check if the access token is valid (i.e. exists in `users` table)
    # If it is, then run the model.
    # return Response(content=wav.getvalue(), media_type="audio/wav")
    # If it isn't, then return a 401.


if __name__ == "__main__":
    stub.serve()
