import os

import fastapi
import fastapi.middleware.cors
import modal
from fastapi import Request
from fastapi.responses import Response

from model import TortoiseModal, stub

## Setup FastAPI server.
web_app = fastapi.FastAPI()
web_app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###### MODAL CPU API Key Logic.

supabase_image = modal.Image.debian_slim().pip_install("supabase")


@web_app.post("/")
def post_request(req: Request):
    """
    POST endpoint for running Tortoise. Checks whether the user exists,
    and adds usage time to the user's account.
    """
    import asyncio
    import time

    from supabase import Client, create_client

    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(url, key)

    body = asyncio.run(req.json())
    text = body["text"]
    voices = body["voices"]
    api_key = body["api_key"]
    target_file_web_paths = body.get("target_file", None)

    data = supabase.table("users").select("*").eq("api_key", api_key).execute().data
    
    # check: out of credits
    if data[0]["usage_dollar"] >= data[0]["max_credit_dollar"]:
        return Response(
            status_code=403,
            content="No more credits left. Please upgrade to the pay-as-you-go plan",
        )

    if target_file_web_paths:
        supabase.table("synthesis").insert(
            {
                "user_id": data[0]["id"],
                "input": text,
                "voices": voices,
                "target_file_web_paths": target_file_web_paths,
            }
        ).execute()
    else:
        supabase.table("synthesis").insert(
            {
                "user_id": data[0]["id"],
                "input": text,
                "voices": voices,
            }
        ).execute()

    if len(data) == 1:
        # user is registered.

        start = time.time()
        wav = TortoiseModal().run_tts.call(text, voices, target_file_web_paths)
        end = time.time()

        # Update the user's usage.
        fresh_data_usage_dollar = (
            supabase.table("users")
            .select("*")
            .eq("api_key", api_key)
            .execute()
            .data[0]["usage_dollar"]
        )
        supabase.table("users").update(
            {"usage_dollar": fresh_data_usage_dollar + 0.0005833 * (end - start)}
        ).eq("api_key", api_key).execute()

        return Response(content=wav.getvalue(), media_type="audio/wav")
    else:
        # return a 401
        return Response(status_code=401)


@stub.asgi(
    image=supabase_image,
    secret=modal.Secret.from_name("supabase-tortoise-secrets"),
)
def app():
    return web_app


if __name__ == "__main__":
    stub.serve()
