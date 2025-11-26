#!/usr/bin/env python3

import asyncio

from openai import AsyncOpenAI


async def test_tool_call(client: AsyncOpenAI, model: str):
    """Test tool calling capability."""

    question = """Two questions, answer quickly:
1. What model are you?
2. What does METR (organization) stand for?
    """

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        tools=[],
        max_tokens=1500,
    )

    message = response.choices[0].message
    print(f"Question:\n\033[96m{question}\033[0m")
    print(f"Reasoning:\n\033[94m{message.reasoning}\033[0m")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    print(f"Result:\n\033[92m{message.content}\033[0m")


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="moonshotai/Kimi-K2-Thinking")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.url, api_key="EMPTY")

    print(f"Connecting to {args.url}\n")

    await test_tool_call(client, args.model)

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
