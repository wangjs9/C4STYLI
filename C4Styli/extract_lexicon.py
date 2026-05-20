import argparse
import json
import multiprocessing
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openai
from tqdm import trange
from utils import *

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", None)

instruction_zh = """你是一个语言学上非常保守的标注员。

给定一批中文文本（标题、广告语等），请你只抽取“单独本身就带有情绪或情感态度”的词或短语，生成一个统一的情绪词表。

标注规则：
- 只抽取单独就能表达情绪、感受或主观态度的词或短语
- 不要抽取程度副词（如：很、非常、极其、特别）
- 不要抽取语气词或功能词（如：啊、呀、吧、呢）
- 不要抽取中性描述性词语
- 如果文本中没有明确的情绪词，不要随意生成

请只输出一个 **合并后的 LIST 列表**，去重即可，不要输出解释。

文本：
{text}
"""

instruction_yue = """你係一個熟悉粵語嘅語言學標注員。

而家畀你一批中文文本（電影標題、廣告語等等），請你只揀出「單獨本身就帶有情緒或者主觀態度」嘅詞語或者短語，生成一個統一嘅情緒詞表。

標注規則：
- 只揀出單獨睇都可以表達情緒、感受或者態度嘅詞語或短語
- 唔好揀程度副詞（例如：好、好鬼、非常、極其）
- 唔好揀語氣助詞或者功能詞（例如：啊、呀、吧、呢、啦、喎、咯、嘛）
- 唔好揀中性描述性詞語
- 如果文本中冇明確嘅情緒詞，唔好亂生成

請只輸出一個 **合併後嘅 LIST 列表**，去重即可，唔好輸出任何解釋。

文本：
{text}
"""


def get_utterance(text, url_base, model_name, prompt, args):
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
                    {"role": "user", "content": prompt.format(text=text)},
                ],
                max_tokens=args.max_new_tokens,
                n=1,
                timeout=300,  # 5 minutes timeout per request
            )
            utterance = response.choices[0].message.content
            if utterance is None or utterance == "":
                print(f"Invalid response: {utterance}")

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
    text, url_base, model_name, prompt, args = params
    return get_utterance(text, url_base, model_name, prompt, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="CN", choices=["CN", "HK"])
    parser.add_argument(
        "--domain", type=str, default="titles", choices=["titles", "slogans"]
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--url_base", type=str, default="https://api.deepseek.com")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    if args.region == "CN" and args.domain == "titles":
        dataset = json.load(
            open("C4Styli/titles/val_movie_titles.json", "r", encoding="utf-8")
        )
        dataset += json.load(
            open("C4Styli/titles/finetune_movie_titles.json", "r", encoding="utf-8")
        )
        dataset += json.load(
            open("C4Styli/titles/probe_movie_titles.json", "r", encoding="utf-8")
        )
        text_list = [item["TITLE (CN)"] for item in dataset]
        prompt = instruction_zh
        # with open("C4Styli/lexicon/titles_CN_lexicon.json", "rt+", encoding="utf-8") as f:
        #     processed_data = json.load(f)
        # processed_data_input_texts = [item["input_text"] for item in processed_data]
        # text_list = [item for item in text_list if item not in processed_data_input_texts]
    elif args.region == "CN" and args.domain == "slogans":
        dataset = json.load(
            open("C4Styli/slogans/val_advertise_slogans.json", "r", encoding="utf-8")
        )
        dataset += json.load(
            open(
                "C4Styli/slogans/finetune_advertise_slogans.json", "r", encoding="utf-8"
            )
        )
        dataset += json.load(
            open("C4Styli/slogans/probe_advertise_slogans.json", "r", encoding="utf-8")
        )
        text_list = [item["slogan"] for item in dataset if item["region"] == "CN"]
        prompt = instruction_zh
        # with open("C4Styli/lexicon/slogans_CN_lexicon.json", "rt+", encoding="utf-8") as f:
        #     processed_data = json.load(f)
        # processed_data_input_texts = [item["input_text"] for item in processed_data]
        # text_list = [item for item in text_list if item not in processed_data_input_texts]
    elif args.region == "HK" and args.domain == "slogans":
        dataset = json.load(
            open("C4Styli/slogans/val_advertise_slogans.json", "r", encoding="utf-8")
        )
        dataset += json.load(
            open(
                "C4Styli/slogans/finetune_advertise_slogans.json", "r", encoding="utf-8"
            )
        )
        dataset += json.load(
            open("C4Styli/slogans/probe_advertise_slogans.json", "r", encoding="utf-8")
        )
        text_list = [item["slogan"] for item in dataset if item["region"] == "HK"]
        prompt = instruction_yue
        # with open("C4Styli/lexicon/slogans_HK_lexicon.json", "rt+", encoding="utf-8") as f:
        #     processed_data = json.load(f)
        # processed_data_input_texts = [item["input_text"] for item in processed_data]
        # text_list = [item for item in text_list if item not in processed_data_input_texts]
    elif args.region == "HK" and args.domain == "titles":
        dataset = json.load(
            open("C4Styli/titles/val_movie_titles.json", "r", encoding="utf-8")
        )
        dataset += json.load(
            open("C4Styli/titles/finetune_movie_titles.json", "r", encoding="utf-8")
        )
        dataset += json.load(
            open("C4Styli/titles/probe_movie_titles.json", "r", encoding="utf-8")
        )
        text_list = [item["TITLE (HK)"] for item in dataset]
        prompt = instruction_yue
        # with open("C4Styli/lexicon/titles_HK_lexicon.json", "rt+", encoding="utf-8") as f:
        #     processed_data = json.load(f)
        # processed_data_input_texts = [item["input_text"] for item in processed_data]
        # text_list = [item for item in text_list if item not in processed_data_input_texts]
    else:
        raise ValueError(f"Invalid region: {args.region}")

    processed_data = []
    for i in trange(0, len(text_list), args.batch_size, desc="Extracting lexicon"):
        batch_text_list = text_list[i : i + args.batch_size]
        with multiprocessing.get_context("spawn").Pool(
            processes=min(8, len(batch_text_list))
        ) as pool:
            outputs = pool.map(
                get_batch_utterances,
                [
                    (text, args.url_base, args.model, prompt, args)
                    for text in batch_text_list
                ],
            )
        for text, lexicon in zip(batch_text_list, outputs):
            processed_data.append(
                {
                    "input_text": text,
                    "output_text": lexicon,
                }
            )
        if not os.path.exists(f"C4Styli/lexicon"):
            os.makedirs(f"C4Styli/lexicon")
        with open(
            f"C4Styli/lexicon/{args.domain}_{args.region}_lexicon.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(
        f"Saved {len(processed_data)} processed data to C4Styli/lexicon/{args.domain}_{args.region}_lexicon.json"
    )
