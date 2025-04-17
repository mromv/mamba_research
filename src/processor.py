import os
import random
import argparse
import httpx
import asyncio

from typing import Dict
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from .utils import (
    load_config,
    log_error,
    save_sample
)

load_dotenv()
API_KEY = os.environ["DEEPSEEK_API_KEY"]

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

URL = "https://api.deepseek.com/v1/chat/completions"
semaphore = asyncio.Semaphore(16)

async def get_response_async(
    session: httpx.AsyncClient,
    query: str,
    sample: Dict,
    idx: int,
    system_prompt: str,
    retries: int = 3,
    backoff_base: float = 1.5
):
    async with semaphore:
        data = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "max_tokens": 32,
            "stream": False
        }

        for attempt in range(retries):
            try:
                resp = await session.post(URL, headers=HEADERS, json=data, timeout=600)
                if resp.status_code == 200:
                    result = resp.json()
                    answer = result['choices'][0]['message']['content']
                    reasoning = result['choices'][0]['message'].get('reasoning_content', "")
                    return answer, reasoning
                elif resp.status_code in {429, 500, 502, 503, 504}:
                    wait_time = backoff_base ** attempt + random.uniform(0, 0.5)
                    await asyncio.sleep(wait_time)
                else:
                    error_msg = f"HTTP {resp.status_code}: {resp.text}"
                    await log_error(idx, sample, error_msg)
                    return None
            except Exception as e:
                wait_time = backoff_base ** attempt + random.uniform(0, 0.5)
                await asyncio.sleep(wait_time)

        final_error = "Error: Retry limit exceeded"
        await log_error(idx, sample, final_error)
        return None

async def process_sample(
    idx: int,
    sample: Dict,
    session: httpx.AsyncClient,
    output_file: str,
    lock: asyncio.Lock,
    template: str,
    prompt: str
):
    query = template.format(**sample)
    result = await get_response_async(session, query, sample, idx, prompt)
    if result is None:
        return

    answer, reasoning = result
    sample.update({'answer': answer, 'reasoning': reasoning})
    await save_sample(sample, output_file, lock)

async def process_dataset(dataset_name: str):
    config = load_config(dataset_name)
    split = config["dataset_config"].get("split", "train")
    
    dataset = load_dataset(
        config["dataset_config"]["name"],
        config["dataset_config"].get("subset"),
        split=split,
    )

    output_file = f"data/{dataset_name}_{split}.jsonl"
    
    async with httpx.AsyncClient() as session:
        lock = asyncio.Lock()
        tasks = [
            asyncio.create_task(
                process_sample(
                    idx, sample, session, output_file, lock,
                    config["template"], config["prompt"]
                )
            )
            for idx, sample in enumerate(dataset)
        ]
        
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            await task

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    
    asyncio.run(process_dataset(args.dataset))
