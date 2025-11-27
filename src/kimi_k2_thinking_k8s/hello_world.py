#!/usr/bin/env python3

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)


@dataclass
class ToolCallResult:
    """Result of a complete tool calling interaction."""

    response: ChatCompletion
    random_number: int | None
    final_answer: str


def get_random_number() -> int:
    """Returns a random number between 1 and 100."""
    return random.randint(1, 100)


def execute_tool(tool_name: str, _arguments: dict[str, Any]) -> Any:
    """Execute a tool by name with given arguments."""
    if tool_name == "get_random_number":
        return get_random_number()
    raise ValueError(f"Unknown tool: {tool_name}")


async def chat(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int = 1500,
):
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    # Step 1: Initial request
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    response_message = response.choices[0].message
    return response_message


async def run_parallel_requests(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    n_requests: int,
    max_concurrent: int,
):
    """Run multiple chat requests in parallel with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_request():
        async with semaphore:
            return await chat(client, prompt, model)

    return await asyncio.gather(*[bounded_request() for _ in range(n_requests)])


def print_results(messages: list[ChatCompletionMessage], elapsed_time: float) -> None:
    """Print formatted results from the tool calling interactions."""
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}\n")

    for message in messages[:5]:
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            print(f"  Reasoning: \033[94m{reasoning}\033[0m")

        print(f"  Answer: \033[92m{message.content}\033[0m\n")

    print(f"{'=' * 80}")
    print(f"Total requests: {len(messages)}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Average time per request: {elapsed_time / len(messages):.2f}s")
    print(f"{'=' * 80}")


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Test vLLM server with tool calling capabilities"
    )
    parser.add_argument(
        "--api-key",
        help="API key",
    )
    parser.add_argument(
        "--url", default="http://metr.romaingrx.com:30080/v1", help="API base URL"
    )
    parser.add_argument(
        "--model", default="moonshotai/kimi-k2-thinking", help="Model name"
    )
    parser.add_argument(
        "--n-requests", type=int, default=1, help="Number of parallel requests"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10, help="Max concurrent requests"
    )
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.url, api_key=args.api_key)

    print(f"Connecting to {args.url}")
    print(f"Model: {args.model}")
    print(
        f"Running {args.n_requests} requests with max {args.max_concurrent} concurrent\n"
    )

    prompt = (
        "Give me the answer to the following question: What is the capital of France?"
    )

    start = time.time()
    results = await run_parallel_requests(
        client=client,
        prompt=prompt,
        model=args.model,
        n_requests=args.n_requests,
        max_concurrent=args.max_concurrent,
    )
    elapsed = time.time() - start

    print_results(results, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
