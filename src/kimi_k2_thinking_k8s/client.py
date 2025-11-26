#!/usr/bin/env python3

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolParam,
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


async def chat_with_tools(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int = 1500,
) -> ToolCallResult:
    """
    Send a chat request with tool calling support.

    Handles the complete tool calling flow:
    1. Send initial request
    2. Execute any tool calls
    3. Send tool results back for final response

    Returns:
        ToolCallResult containing the final response and execution details
    """
    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_random_number",
                "description": "Returns a random number between 1 and 100",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    # Step 1: Initial request
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=max_tokens,
    )

    response_message = response.choices[0].message
    random_number = None
    tool_calls = cast(
        list[ChatCompletionMessageFunctionToolCall], response_message.tool_calls
    )

    # Step 2: Handle tool calls if any
    if tool_calls:
        # Build properly typed tool call params
        tool_call_params: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]

        # Append the assistant's response (including tool_calls)
        messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": tool_call_params,
            }
        )

        # Process each tool call
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args: dict[str, Any] = (
                json.loads(tool_call.function.arguments)
                if tool_call.function.arguments
                else {}
            )

            # Execute the tool
            result = execute_tool(tool_name, tool_args)
            if tool_name == "get_random_number":
                random_number = result

            # Step 3: Append tool result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"number": result}),
                }
            )

        # Step 4: Get final response
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )

    return ToolCallResult(
        response=response,
        random_number=random_number,
        final_answer=response.choices[0].message.content or "",
    )


async def run_parallel_requests(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    n_requests: int,
    max_concurrent: int,
) -> list[ToolCallResult]:
    """Run multiple chat requests in parallel with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_request() -> ToolCallResult:
        async with semaphore:
            return await chat_with_tools(client, prompt, model)

    return await asyncio.gather(*[bounded_request() for _ in range(n_requests)])


def print_results(results: list[ToolCallResult], elapsed_time: float) -> None:
    """Print formatted results from the tool calling interactions."""
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}\n")

    for i, result in enumerate(results[:5], 1):
        message = result.response.choices[0].message

        print(f"Request {i}:")
        if result.random_number is not None:
            print(f"  Random number: \033[93m{result.random_number}\033[0m")

        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            print(f"  Reasoning: \033[94m{reasoning}\033[0m")

        print(f"  Answer: \033[92m{result.final_answer}\033[0m\n")

    print(f"{'=' * 80}")
    print(f"Total requests: {len(results)}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Average time per request: {elapsed_time / len(results):.2f}s")
    print(f"{'=' * 80}")


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Test vLLM server with tool calling capabilities"
    )
    parser.add_argument(
        "--url", default="http://localhost:8000/v1", help="API base URL"
    )
    parser.add_argument(
        "--model", default="moonshotai/Kimi-K2-Thinking", help="Model name"
    )
    parser.add_argument(
        "--n-requests", type=int, default=10, help="Number of parallel requests"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10, help="Max concurrent requests"
    )
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.url, api_key="EMPTY")

    print(f"Connecting to {args.url}")
    print(f"Model: {args.model}")
    print(
        f"Running {args.n_requests} requests with max {args.max_concurrent} concurrent\n"
    )

    prompt = "Please use the get_random_number tool to get a random number, then multiply it by 2 and tell me the result."

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
