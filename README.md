# 🐢 Tortoise TTS pay-as-you-go API

A "pay-as-you-go" API for [Tortoise TTS](https://github.com/neonbjb/tortoise-tts/). It uses [Modal](https://modal.com/) underneath.

Tortoise is one of the best text-to-speech systems ever built, but it currently requires someone to deploy their own service on a GPU which can be time-consuming and/or difficult. The other alternative is to use similar paid services such as play.ht which offer a monthly pricing and are closed-off. This repository aims to provide a usage-based pay-as-you-go API based on the open-source code instead.

We have made some improvements to Tortoise to make the inference 30% faster, and welcome contributions to improve it further!

## Developer environment
- `python=3.10.8` (important for Modal)
