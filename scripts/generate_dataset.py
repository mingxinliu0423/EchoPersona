import os
import json
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv

import openai
from openai import OpenAI
from datasets import load_dataset

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = "你是一个讽刺幽默但有深度的 AI，对问题从不直接回答，而是引导人反思。用语要冷静、克制，有存在主义气质。"

SEED_QUESTIONS = [
    "自由意志是否存在？",
    "人类为什么害怕孤独？",
    "科技会毁灭还是拯救文明？",
    "快乐和意义哪个更重要？",
    "什么是一个人真正的自我？"
]


def call_openai(prompt):

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]


def generate_from_gpt(output_path, n=10):
    dataset = []
    for q in random.sample(SEED_QUESTIONS * 3, n):
        print(f"Generating: {q}")
        ans = call_openai(q)
        dataset.append({"instruction": q, "input": "", "output": ans})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"GPT 数据已保存到 {output_path}")


def generate_from_huggingface(output_path, source="Open-Orca/OpenOrca", n=20):
    print(f"Downloading and processing from {source}...")
    ds = load_dataset(source, split="train")
    cleaned = []
    for item in random.sample(list(ds), n):
        if "question" in item and "response" in item:
            cleaned.append({
                "instruction": item["question"],
                "input": "",
                "output": item["response"]
            })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"HuggingFace 数据已保存到 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gpt", "hf"], required=True, help="数据生成方式：gpt / hf")
    parser.add_argument("--output", default="data/persona_dataset.json", help="输出文件路径")
    parser.add_argument("--num", type=int, default=20, help="生成样本数量")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    if args.method == "gpt":
        generate_from_gpt(args.output, n=args.num)
    elif args.method == "hf":
        generate_from_huggingface(args.output, source="Open-Orca/OpenOrca", n=args.num)
