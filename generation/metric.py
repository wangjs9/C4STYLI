import argparse
import json
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

_embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def compare_titles(model_name):
    data_path = f"generation/output/{model_name}_titles.json"
    data = json.load(open(data_path, "r", encoding="utf-8"))

    processed_data = []
    for i in range(0, len(data), 2):
        hk_item = data[i]
        cn_item = data[i + 1]
        region = hk_item["region"]
        title = hk_item["TITLE"]
        title_cn = cn_item["TITLE (CN)"]
        title_hk = hk_item["TITLE (HK)"]
        plot_summary = hk_item["PLOT SUMMARY"]
        if "Qwen" in model_name:
            # 这里是去掉thinking
            cn_output = re.sub(
                r"<think>.*?</think>", "", cn_item["output_text"], flags=re.DOTALL
            ).strip()
            hk_output = re.sub(
                r"<think>.*?</think>", "", hk_item["output_text"], flags=re.DOTALL
            ).strip()
            cn_thinking_matches = re.findall(
                r"<think>(.*?)</think>", cn_item["output_text"], re.DOTALL
            )
            cn_thinking = cn_thinking_matches[0].strip() if cn_thinking_matches else ""
            hk_thinking_matches = re.findall(
                r"<think>(.*?)</think>", hk_item["output_text"], re.DOTALL
            )
            hk_thinking = hk_thinking_matches[0].strip() if hk_thinking_matches else ""
        else:
            cn_output = cn_item["output_text"]
            hk_output = hk_item["output_text"]
            cn_thinking = ""
            hk_thinking = ""

        processed_data.append(
            {
                "title": title,
                "year": hk_item["YEAR"],
                "plot_summary": plot_summary,
                "title_cn": title_cn,
                "title_hk": title_hk,
                "LLM-generated CN title": cn_output,
                "LLM-generated HK title": hk_output,
                "LLM-generated CN thinking": cn_thinking,
                "LLM-generated HK thinking": hk_thinking,
            }
        )

    results = []
    for item in processed_data:
        title = item["title"]
        plot_summary = item["plot_summary"]
        title_cn = item["title_cn"]
        title_hk = item["title_hk"]
        LLM_generated_CN_title = item["LLM-generated CN title"]
        LLM_generated_HK_title = item["LLM-generated HK title"]

        cn_preservation = calc_similarity(title_cn, LLM_generated_CN_title)
        hk_preservation = calc_similarity(title_hk, LLM_generated_HK_title)
        distinction = 1 - calc_similarity(
            LLM_generated_HK_title, LLM_generated_CN_title
        )
        original_distinction = 1 - calc_similarity(title_hk, title_cn)
        results.append(
            {
                "region": "CN",
                "distinction": distinction,
                "preservation": cn_preservation,
                "original_distinction": original_distinction,
                "BAI": 2
                * (cn_preservation * distinction)
                / (cn_preservation + distinction),
            }
        )
        results.append(
            {
                "region": "HK",
                "distinction": distinction,
                "preservation": hk_preservation,
                "original_distinction": original_distinction,
                "BAI": 2
                * (hk_preservation * distinction)
                / (hk_preservation + distinction),
            }
        )

    overall = {
        "distinction_rate": sum(item["distinction"] for item in results) / len(results),
        "preservation_rate": sum(item["preservation"] for item in results)
        / len(results),
        "original_distinction_rate": sum(
            item["original_distinction"] for item in results
        )
        / len(results),
        "BAI_rate": sum(item["BAI"] for item in results) / len(results),
        "total_samples": len(results),
        "valid_samples": len(results),
    }
    CN_overall = {
        "distinction_rate": sum(
            item["distinction"] for item in results if item["region"] == "CN"
        )
        / len([item for item in results if item["region"] == "CN"]),
        "preservation_rate": sum(
            item["preservation"] for item in results if item["region"] == "CN"
        )
        / len([item for item in results if item["region"] == "CN"]),
        "BAI_rate": sum(item["BAI"] for item in results if item["region"] == "CN")
        / len([item for item in results if item["region"] == "CN"]),
        "total_samples": len([item for item in results if item["region"] == "CN"]),
        "valid_samples": len([item for item in results if item["region"] == "CN"]),
    }
    HK_overall = {
        "distinction_rate": sum(
            item["distinction"] for item in results if item["region"] == "HK"
        )
        / len([item for item in results if item["region"] == "HK"]),
        "preservation_rate": sum(
            item["preservation"] for item in results if item["region"] == "HK"
        )
        / len([item for item in results if item["region"] == "HK"]),
        "BAI_rate": sum(item["BAI"] for item in results if item["region"] == "HK")
        / len([item for item in results if item["region"] == "HK"]),
        "total_samples": len([item for item in results if item["region"] == "HK"]),
        "valid_samples": len([item for item in results if item["region"] == "HK"]),
    }
    result_json = {
        "overall": overall,
        "CN_overall": CN_overall,
        "HK_overall": HK_overall,
        "results": results,
    }
    return processed_data, result_json


def compare_slogans(model_name):
    data_path = f"generation/output/{model_name}_slogans.json"
    data = json.load(open(data_path, "r", encoding="utf-8"))

    processed_data = []
    for i in range(0, len(data), 2):
        reverse = data[i]
        original = data[i + 1]

        company = original["company"]
        product = original["product"]
        year = original.get("year", -1)
        region = original["region"]
        slogan = original["slogan"]
        domain = original["domain"]

        original_output = original["output_text"]
        reverse_output = reverse["output_text"]
        original_thinking = original["output_text"].replace("<think>.*?</think>", "")
        reverse_thinking = reverse["output_text"].replace("<think>.*?</think>", "")
        original_thinking_matches = re.findall(
            r"<think>(.*?)</think>", original["output_text"], re.DOTALL
        )
        original_thinking = (
            original_thinking_matches[0].strip() if original_thinking_matches else ""
        )
        reverse_thinking_matches = re.findall(
            r"<think>(.*?)</think>", reverse["output_text"], re.DOTALL
        )
        reverse_thinking = (
            reverse_thinking_matches[0].strip() if reverse_thinking_matches else ""
        )

        processed_data.append(
            {
                "company": company,
                "product": product,
                "year": year,
                "region": region,
                "slogan": slogan,
                "domain": domain,
                "LLM-generated CN slogan": original_output
                if region == "CN"
                else reverse_output,
                "LLM-generated HK slogan": reverse_output
                if region == "CN"
                else original_output,
                "LLM-generated CN thinking": original_thinking
                if region == "CN"
                else reverse_thinking,
                "LLM-generated HK thinking": reverse_thinking
                if region == "CN"
                else original_thinking,
            }
        )

    results = []
    for item in processed_data:
        region = item["region"]
        slogan = item["slogan"]

        distinction = 1 - calc_similarity(
            item["LLM-generated HK slogan"], item["LLM-generated CN slogan"]
        )
        if region == "CN":
            same_region_slogan = item["LLM-generated CN slogan"]
        else:
            same_region_slogan = item["LLM-generated HK slogan"]
        preservation = calc_similarity(same_region_slogan, slogan)
        results.append(
            {
                "region": region,
                "distinction": distinction,
                "preservation": preservation,
                "BAI": 2 * (preservation * distinction) / (preservation + distinction),
            }
        )
    overall = {
        "distinction_rate": sum(item["distinction"] for item in results) / len(results),
        "preservation_rate": sum(item["preservation"] for item in results)
        / len(results),
        "BAI_rate": sum(item["BAI"] for item in results) / len(results),
        "total_samples": len(results),
        "valid_samples": len(results),
    }

    CN_overall = {
        "distinction_rate": sum(
            item["distinction"] for item in results if item["region"] == "CN"
        )
        / len([item for item in results if item["region"] == "CN"]),
        "preservation_rate": sum(
            item["preservation"] for item in results if item["region"] == "CN"
        )
        / len([item for item in results if item["region"] == "CN"]),
        "BAI_rate": sum(item["BAI"] for item in results if item["region"] == "CN")
        / len([item for item in results if item["region"] == "CN"]),
        "total_samples": len([item for item in results if item["region"] == "CN"]),
        "valid_samples": len([item for item in results if item["region"] == "CN"]),
    }
    HK_overall = {
        "distinction_rate": sum(
            item["distinction"] for item in results if item["region"] == "HK"
        )
        / len([item for item in results if item["region"] == "HK"]),
        "preservation_rate": sum(
            item["preservation"] for item in results if item["region"] == "HK"
        )
        / len([item for item in results if item["region"] == "HK"]),
        "BAI_rate": sum(item["BAI"] for item in results if item["region"] == "HK")
        / len([item for item in results if item["region"] == "HK"]),
        "total_samples": len([item for item in results if item["region"] == "HK"]),
        "valid_samples": len([item for item in results if item["region"] == "HK"]),
    }

    result_json = {
        "overall": overall,
        "CN_overall": CN_overall,
        "HK_overall": HK_overall,
        "results": results,
    }

    return processed_data, result_json


def calc_similarity(text_a, text_b):
    """计算语义相似度（基于embedding）"""
    if not text_a or not text_b:
        return 0.0

    embeddings = _embedder.encode([text_a, text_b])
    vec_a, vec_b = embeddings[0], embeddings[1]

    similarity = 1 - cosine(vec_a, vec_b)
    return float(similarity)


def compare_similarity(domain="slogans"):
    if domain == "slogans":
        data_path = "C4Styli/slogans/slogan_pairs.json"
        data = json.load(open(data_path, "r", encoding="utf-8"))
        CN_data = [item["CN"]["slogan"] for item in data]
        HK_data = [item["HK_matches"][0]["slogan"] for item in data]

    elif domain == "titles":
        data_path = "C4Styli/titles/val_movie_titles.json"
        data = json.load(open(data_path, "r", encoding="utf-8"))
        CN_data = [item["TITLE (CN)"] for item in data]
        HK_data = [item["TITLE (HK)"] for item in data]

    CN_embeddings = _embedder.encode(CN_data)  # (286, 768)
    HK_embeddings = _embedder.encode(HK_data)  # (175, 768)

    # 计算每一对 CN-HK 配对的语义相似度并进行分布统计
    pair_similarities = []
    for cn_emb, hk_emb in zip(CN_embeddings, HK_embeddings):
        sim = 1 - cosine(cn_emb, hk_emb)
        pair_similarities.append(float(sim))
    pair_similarities = np.array(pair_similarities)

    print(f"相似度分布统计：")
    print(f"  均值: {pair_similarities.mean():.4f}")
    print(f"  标准差: {pair_similarities.std():.4f}")
    print(f"  最大值: {pair_similarities.max():.4f}")
    print(f"  最小值: {pair_similarities.min():.4f}")

    # 可视化分布
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(pair_similarities, bins=30, alpha=0.8, color="skyblue", edgecolor="k")
    plt.title("CN-HK Pairwise Similarity Distribution")
    plt.xlabel("Semantic Similarity")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{domain}_pairwise_similarity_distribution.png", dpi=300)
    plt.show()


def visualize_distribution(domain="slogans"):
    if domain == "slogans":
        data_path = "C4Styli/slogans/val_advertise_slogans.json"
        data = json.load(open(data_path, "r", encoding="utf-8"))
        CN_data = [item["slogan"] for item in data if item["region"] == "CN"]
        HK_data = [item["slogan"] for item in data if item["region"] == "HK"]
    elif domain == "titles":
        data_path = "C4Styli/titles/val_movie_titles.json"
        data = json.load(open(data_path, "r", encoding="utf-8"))
        CN_data = [item["TITLE (CN)"] for item in data]
        HK_data = [item["TITLE (HK)"] for item in data]
    else:
        raise ValueError(f"Invalid domain: {domain}")
    CN_embeddings = _embedder.encode(CN_data)  # (286, 768)
    HK_embeddings = _embedder.encode(HK_data)  # (175, 768)

    pca = PCA(n_components=2)
    all_embeddings = np.vstack([CN_embeddings, HK_embeddings])
    all_pca = pca.fit_transform(all_embeddings)

    CN_pca = all_pca[:286]
    HK_pca = all_pca[286:]

    # 计算两个分布的距离 (Wasserstein distance)
    dist_dim1 = wasserstein_distance(CN_pca[:, 0], HK_pca[:, 0])
    dist_dim2 = wasserstein_distance(CN_pca[:, 1], HK_pca[:, 1])

    print(f"Distribution distance (dim 1): {dist_dim1:.3f}")
    print(f"Distribution distance (dim 2): {dist_dim2:.3f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(CN_pca[:, 0], CN_pca[:, 1], alpha=0.5, label="CN", s=30)
    plt.scatter(HK_pca[:, 0], HK_pca[:, 1], alpha=0.5, label="HK", s=30)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("CN vs HK Slogan Distributions")
    plt.savefig("cn_hk_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-32B")
    parser.add_argument(
        "--domain", type=str, default="titles", choices=["titles", "slogans"]
    )
    args = parser.parse_args()
    if args.domain == "titles":
        data, results = compare_titles(args.model_name)
    elif args.domain == "slogans":
        data, results = compare_slogans(args.model_name)
    else:
        raise ValueError(f"Invalid domain: {args.domain}")

    json.dump(
        data,
        open(
            f"generation/output/{args.model_name}_{args.domain}_pairs.json",
            "w",
            encoding="utf-8",
        ),
        ensure_ascii=False,
        indent=2,
    )

    print(results["overall"])
    print(results["CN_overall"])
    print(results["HK_overall"])
    with open(
        f"generation/output/{args.model_name}_{args.domain}_metric.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # compare_similarity(domain="titles")
    main()
