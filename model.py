import io
import os
import subprocess
import tempfile

import modal

stub = modal.Stub("tts")

tortoise_image = (
    modal.Image.conda()
    .apt_install("git")
    .apt_install("libsndfile-dev")
    .apt_install("ffmpeg")
    .apt_install("curl")
    .run_commands(
        "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116",
        "git clone https://github.com/vatsalaggarwal/tortoise-tts",
        "cd tortoise-tts; pip install -r requirements.txt; pip install -e .",
        # download models as part of the container.
        "python -c 'from tortoise.api import MODELS_DIR, TextToSpeech;tts = TextToSpeech(models_dir=MODELS_DIR); tts.get_random_conditioning_latents()'",
        "pip install pydub",
    )
)


class TortoiseModal:
    def __enter__(self):
        """
        Load the model weights into GPU memory when the container starts.
        """
        from tortoise.api import MODELS_DIR, TextToSpeech
        from tortoise.utils.audio import load_audio, load_voices

        self.load_voices = load_voices
        self.load_audio = load_audio
        self.tts = TextToSpeech(models_dir=MODELS_DIR)
        self.tts.get_random_conditioning_latents()

    def process_synthesis_result(self, result):
        """
        Converts a audio torch tensor to a binary blob.
        """
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

    def load_target_file(self, target_file_web_path, name):
        """
        Downloads a target file from a static file store web and stores it in a directory structure
        expected by Tortoise.

        All new voices are stored in /voices/, and the file is downloaded and stored to
        /voices/<name>/<filename>.
        """
        # curl to download file to temp file
        os.makedirs(f"/voices/{name}", exist_ok=True)
        target_file = "/voices/" + f"{name}/" + os.path.split(target_file_web_path)[-1]
        if (
            subprocess.run(
                f"curl -o {target_file} {target_file_web_path}",
                shell=True,
                stdout=subprocess.PIPE,
            ).returncode
            != 0
        ):
            raise ValueError("Failed to download file.")

        # check size -- should be <= 100 Mb
        if os.path.getsize(target_file) > 100000000:
            raise ValueError("File too large.")

        return f"/voices/"

    # TODO: check if you want to use different GPUs?
    @stub.function(image=tortoise_image, gpu="A10G")
    def run_tts(self, text, voices, target_file_web_path):
        """
        Runs tortoise tts on a given text and voice. Alternatively, a
        web path can be to a target file to be used instead of a voice for
        one-shot synthesis.
        """
        CANDIDATES = 1  # NOTE: this code only works for one candidate.
        CVVP_AMOUNT = 0.0
        SEED = None
        PRESET = "fast"

        if target_file_web_path is not None:
            voice_name = "target"
            if voices != "":
                raise ValueError("Cannot specify both target_file and voices.")
            target_dir = self.load_target_file(target_file_web_path, name=voice_name)
            voice_samples, conditioning_latents = self.load_voices(
                [voice_name], extra_voice_dirs=[target_dir]
            )
        else:
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
