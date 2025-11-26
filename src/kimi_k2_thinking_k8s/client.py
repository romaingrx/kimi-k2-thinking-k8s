#!/usr/bin/env python3

import asyncio
import time

from openai import AsyncOpenAI


async def test_tool_call(
    client: AsyncOpenAI, question: str, model: str, semaphore: asyncio.Semaphore
):
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            tools=[],
            max_tokens=1500,
        )
    return response


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="moonshotai/Kimi-K2-Thinking")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.url, api_key="EMPTY")

    print(f"Connecting to {args.url}\n")

    n_parallel = 50
    semaphore = asyncio.Semaphore(n_parallel)

    question = """Two questions, answer quickly:
1. What model are you?
2. What does METR (organization) stand for?
    """

    start = time.time()
    responses = await asyncio.gather(
        *[
            test_tool_call(client, question, args.model, semaphore)
            for _ in range(n_parallel)
        ]
    )
    stop = time.time()

    for response in responses[:5]:
        message = response.choices[0].message
        print(f"Question:\n\033[96m{question}\033[0m")
        print(f"Reasoning:\n\033[94m{message.reasoning}\033[0m")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        print(f"Result:\n\033[92m{message.content}\033[0m")

    print(f"Time taken: {stop - start:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
