#!/usr/bin/env python3
"""
High-concurrency stress test for vLLM to saturate GPUs and engage tensor cores.
"""

import argparse
import asyncio
import os
import time
from collections.abc import Coroutine
from typing import Any, TypedDict

from openai import AsyncOpenAI


class StressTestResponse(TypedDict):
    worker_id: int
    requests: int
    tokens: int
    duration: float


async def send_continuous_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
    duration: int,
) -> StressTestResponse:
    """Send requests continuously for specified duration."""
    start_time = time.time()
    request_count = 0
    total_tokens = 0

    while time.time() - start_time < duration:
        req_start = time.time()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            req_latency = time.time() - req_start
            tokens = response.usage.completion_tokens if response.usage else 0

            request_count += 1
            total_tokens += tokens

            if request_count % 10 == 0:
                print(
                    f"[Worker {request_id:03d}] {request_count} requests, {total_tokens} tokens, avg latency: {req_latency:.2f}s"
                )

        except Exception as e:
            print(f"[Worker {request_id:03d}] Error: {str(e)[:100]}")
            await asyncio.sleep(1)  # Brief pause on error

    return {
        "worker_id": request_id,
        "requests": request_count,
        "tokens": total_tokens,
        "duration": time.time() - start_time,
    }


async def run_stress_test(
    url: str,
    model: str,
    api_key: str,
    num_workers: int,
    duration: int,
    prompt: str,
    max_tokens: int,
):
    """Run continuous stress test with multiple workers."""
    print(f"\n{'=' * 80}")
    print("🔥 STRESS TEST - CONTINUOUS LOAD")
    print(f"{'=' * 80}")
    print(f"URL:              {url}")
    print(f"Concurrent workers: {num_workers}")
    print(f"Duration:         {duration} seconds")
    print(f"Max tokens/req:   {max_tokens}")
    print(f"Prompt:           {prompt[:50]}...")
    print("\nStarting continuous load... Watch your GPU metrics!\n")

    client = AsyncOpenAI(base_url=url, api_key=api_key)

    tasks: list[Coroutine[Any, Any, StressTestResponse]] = []
    for i in range(num_workers):
        task = send_continuous_request(client, model, prompt, max_tokens, i, duration)
        tasks.append(task)

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time

    successful = [r for r in results if isinstance(r, dict)]

    if successful:
        total_requests = sum(r["requests"] for r in successful)
        total_tokens = sum(r["tokens"] for r in successful)

        print(f"\n{'=' * 80}")
        print("✅ STRESS TEST RESULTS")
        print(f"{'=' * 80}")
        print(f"Duration:           {total_time:.1f}s")
        print(f"Workers:            {num_workers}")
        print(f"Total requests:     {total_requests:,}")
        print(f"Total tokens:       {total_tokens:,}")
        print("\nThroughput:")
        print(f"  Requests/sec:     {total_requests / total_time:.2f}")
        print(f"  Tokens/sec:       {total_tokens / total_time:.2f}")
        print(f"  Avg latency:      {total_time / total_requests:.2f}s per request")
        print("\nPer-worker stats:")
        for r in successful[:5]:  # Show first 5 workers
            print(
                f"  Worker {r['worker_id']:03d}: {r['requests']} reqs, {r['tokens']:,} tokens"
            )
        print(f"{'=' * 80}\n")


def main():
    api_key = os.getenv("API_KEY")

    parser = argparse.ArgumentParser(
        description="Continuous stress test for vLLM to saturate GPUs"
    )
    parser.add_argument(
        "--api-key",
        default=api_key,
        required=not api_key,
        help="API key",
    )
    parser.add_argument(
        "--url",
        default="http://metr.romaingrx.com:30080/v1",
        help="vLLM API URL (default: http://metr.romaingrx.com:30080/v1)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=50,
        help="Number of concurrent workers (default: 50)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=120,
        help="Test duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per request (default: 512)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the theory of relativity and its implications for modern physics.",
        help="Prompt to use for all requests",
    )
    parser.add_argument(
        "--model",
        default="moonshotai/kimi-k2-thinking",
        help="Model to use for all requests",
    )

    args = parser.parse_args()

    asyncio.run(
        run_stress_test(
            args.url,
            args.model,
            args.api_key,
            args.workers,
            args.duration,
            args.prompt,
            args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
