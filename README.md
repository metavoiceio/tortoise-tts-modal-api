# 🐢 Tortoise TTS pay-as-you-go API

A "pay-as-you-go" API for [Tortoise TTS](https://github.com/neonbjb/tortoise-tts/). It uses [Modal](https://modal.com/) underneath.

Tortoise is one of the best text-to-speech systems ever built, but it currently requires someone to deploy their own service on a GPU which can be time-consuming and/or difficult. The other alternative is to use similar paid services such as play.ht which offer a monthly pricing and are closed-off. This repository aims to provide a usage-based pay-as-you-go API based on the open-source code instead.

We have made some improvements to Tortoise to make the inference 30% faster, and welcome contributions to improve it further!

## Voices
There are four ways to get different voices out of this model:
- `random`: The model randomly picks a voice in its embedding space
- `<name>`: Use one of the voices the model was trained on (e.g. `train_grace`)
- `<name>&<name>`: Combine two voices (e.g. `train_grace&emma`)
- Zero-shot: Provide a few utterances from a speaker you're trying to clone. This can be used with this repository by uploading those utterances to a static file store (like public S3 bucket) and providing the links to the files within the body of the request (`target_path` parameter).
## Developer environment
- `python=3.10.8` (important for Modal)
