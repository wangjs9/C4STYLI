import argparse
import json
import multiprocessing
import os
import sys
import time

import sensenova

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openai
from tqdm import trange
from utils import *

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "no-api-key-required")


def get_sense_utterance(text, url_base, model_name, args):
    sensenova.access_key_id = os.getenv("SENSE_ACCESS_KEY_ID")
    sensenova.secret_access_key = os.getenv("SENSE_SECRET_ACCESS_KEY")
    utterance = ""
    max_retries = 30
    retry_count = 0
    while utterance == "" or utterance is None:
        try:
            response = sensenova.ChatCompletion.create(
                messages=[
                    {"role": "system", "content": "请根据描述完成相应任务。"},
                    {"role": "user", "content": text},
                ],
                stream=False,
                model=model_name,
                max_new_tokens=1024,
                n=1,
                repetition_penalty=1.05,
                temperature=0.8,
                top_p=0.7,
                know_ids=[],
                user="sensenova-python-test-user",
                # knowledge_config={
                #     "control_level": "normal",
                #     "knowledge_base_result": True,
                #     "knowledge_base_configs": [],
                # },
                # plugins={
                #     "associated_knowledge": {"content": "需要注入给模型的知识", "mode": "concatenate"},
                #     "web_search": {"search_enable": True, "result_enable": True},
                # },
            )
            utterance = response["data"]["choices"][0].get("message")
            return utterance
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            print(f"Error (attempt {retry_count}/{max_retries}): {error_msg}")

            if retry_count >= max_retries:
                print(f"[ERROR] Max retries reached for text: {text[:50]}...")
                utterance = ""  # Return empty string after max retries
                break

            # Exponential backoff: wait 2^retry_count seconds
            wait_time = 2**retry_count
            print(f"[INFO] Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    return utterance


def get_openai_utterance(text, url_base, model_name, args):
    utterance = ""
    client = openai.OpenAI(
        base_url=url_base,  # Point to local server
        api_key=os.getenv("OPENAI_API_KEY"),  # Dummy key for local server
        timeout=60.0,  # 5 minutes timeout for the client
    )
    max_retries = 30
    retry_count = 0

    while utterance == "" or utterance is None:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "请根据描述完成相应任务。",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=args.max_new_tokens,
                n=1,
                timeout=300,  # 5 minutes timeout per request
            )
            utterance = response.choices[0].message.content

        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            print(f"Error (attempt {retry_count}/{max_retries}): {error_msg}")

            if retry_count >= max_retries:
                print(f"[ERROR] Max retries reached for text: {text[:50]}...")
                utterance = ""  # Return empty string after max retries
                break

            # Exponential backoff: wait 2^retry_count seconds
            wait_time = 2**retry_count
            print(f"[INFO] Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    return utterance


def get_batch_utterances(params):
    text, url_base, model_name, args = params
    if "SenseChat" in model_name:
        return get_sense_utterance(text, url_base, model_name, args)
    else:
        return get_openai_utterance(text, url_base, model_name, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--text_domain", type=str, default="titles", choices=["titles", "slogans"]
    )
    parser.add_argument("--url_base", type=str, default="http://127.0.0.1:10086/v1")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_file", type=str, default="output/generation.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prompt_region", type=str, default="CN", choices=["CN", "HK"])
    parser.add_argument("--zero", action="store_true", help="Use zero-shot generation")
    args = parser.parse_args()

    processed_data = []
    # if os.path.exists(args.output_file):
    #     processed_data = json.load(open(args.output_file, "r", encoding="utf-8"))

    if args.text_domain == "titles":
        data = load_movie_data(
            args.dataset_path,
            classification=False,
            prompt_region=args.prompt_region,
            zero=args.zero,
        )
    elif args.text_domain == "slogans":
        data = load_advertising_data(
            args.dataset_path,
            classification=False,
            prompt_region=args.prompt_region,
            zero=args.zero,
        )
    else:
        raise ValueError(f"Invalid text domain: {args.text_domain}")

    processed_data_input_texts = [item["input_text"] for item in processed_data]
    data = [
        item for item in data if item["input_text"] not in processed_data_input_texts
    ]

    if len(data) == 0:
        print(f"[INFO] No data to process")
        exit(0)
    print(f"[INFO] {len(data)} data to process")

    for i in trange(0, len(data), args.batch_size, desc="Processing data"):
        batch_data = data[i : i + args.batch_size]

        with multiprocessing.get_context("spawn").Pool(
            processes=min(8, len(batch_data))
        ) as pool:
            outputs = pool.map(
                get_batch_utterances,
                [
                    (d["input_text"], args.url_base, args.model, args)
                    for d in batch_data
                ],
            )

        for item, o in zip(batch_data, outputs):
            item["output_text"] = o
            processed_data.append(item.copy())
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Saved {len(processed_data)} processed data to {args.output_file}")


if __name__ == "__main__":
    main()
