import os

import openai
from dotenv import load_dotenv

load_dotenv()


def inference_instruct_model(prompt: str) -> str:
    client = openai.OpenAI(
        api_key=os.getenv("HYPERBOLIC_API_KEY"),
        base_url="https://api.hyperbolic.xyz/v1",
    )

    chat_completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct",
        messages=[
            {"role": "system", "content": "Answer with only True or False."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1,
    )

    return chat_completion.choices[0].message.content


def inference_base_model(prompt: str, temperature: float = 0) -> dict[str, float]:
    client = openai.OpenAI(
        api_key=os.getenv("HYPERBOLIC_API_KEY"),
        base_url="https://api.hyperbolic.xyz/v1",
    )

    completion = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B",
        prompt=prompt,
        max_tokens=1,
        logprobs=20,
        temperature=temperature,
    )

    return completion.choices[0].logprobs.top_logprobs[0]


def log_props_base_model(prompt: str, temperature: float = 0) -> dict[str, float]:
    client = openai.OpenAI(
        api_key=os.getenv("HYPERBOLIC_API_KEY"),
        base_url="https://api.hyperbolic.xyz/v1",
    )

    completion = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B",
        prompt=prompt,
        max_tokens=0,
        echo=True,
        logprobs=1,
        temperature=temperature,
    )

    return completion.choices[0].logprobs.token_logprobs
