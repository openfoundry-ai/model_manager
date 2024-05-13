import httpx
from openai import OpenAI, DefaultHttpxClient


def test_openai_proxy():
    client = OpenAI(
        api_key="test-key",
        base_url="http://127.0.0.1:8000/",
        http_client=DefaultHttpxClient(
            proxy="http://127.0.0.1:8000/",
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        )
    )
    chat_completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Say this is a test",
            },
        ],
    )
    print(chat_completion)
