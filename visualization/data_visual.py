import json
from collections import Counter

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

# 设置中文字体，确保支持中文显示
# 字体优先级依次为"SimHei"（常用于中文显示）、"DejaVu Sans"、"Arial Unicode MS"
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

slogan_path = "C4Styli/slogans/finetune_advertise_slogans.json"
title_path = "C4Styli/titles/finetune_movie_titles.json"

slogan_data = json.load(open(slogan_path, "r", encoding="utf-8"))
title_data = json.load(open(title_path, "r", encoding="utf-8"))

slogan_data += json.load(
    open("C4Styli/slogans/val_advertise_slogans.json", "r", encoding="utf-8")
)
title_data += json.load(
    open("C4Styli/titles/val_movie_titles.json", "r", encoding="utf-8")
)


def word_cloud(data, domain="slogan"):
    """
    分别提取粤语和普通话的词云，并构建对应关系

    Args:
        data: 数据列表，每个item包含slogan和region字段
        domain: 数据域，目前支持"slogan"

    Returns:
        dict: 包含cn_text, hk_text, 和对应关系分析的结果
    """
    cn_texts = []
    hk_texts = []

    for item in data:
        if domain == "slogan":
            text = item.get("slogan", "")
            region = item.get("region", "")
        else:
            text = item.get("title", "")
            region = item.get("region", "")

        if region == "CN":
            cn_texts.append(text)
        elif region == "HK":
            hk_texts.append(text)

    cn_text = " ".join(cn_texts)
    hk_text = " ".join(hk_texts)

    # 分析对应关系
    correspondence_analysis = analyze_correspondence(cn_texts, hk_texts)

    return {
        "cn_text": cn_text,
        "hk_text": hk_text,
        "correspondence": correspondence_analysis,
    }


def analyze_correspondence(cn_texts, hk_texts):
    """
    分析粤语和普通话在指代相似物品时的对应关系

    Args:
        cn_texts: 普通话文本列表
        hk_texts: 粤语文本列表

    Returns:
        dict: 对应关系分析结果
    """
    cn_words = []
    for text in cn_texts:
        cn_words.extend(jieba.cut(text))

    hk_words = []
    for text in hk_texts:
        hk_words.extend(jieba.cut(text))

    # 过滤停用词和单字
    stop_words = set(["的", "了", "和", "是", "在", "有", "这", "那", "为", "对", "与", "及", "或"])
    cn_filtered = [w for w in cn_words if len(w) > 1 and w not in stop_words]
    hk_filtered = [w for w in hk_words if len(w) > 1 and w not in stop_words]

    cn_counter = Counter(cn_filtered)
    hk_counter = Counter(hk_filtered)

    # 找出共同词和独特词
    common_words = set(cn_counter.keys()) & set(hk_counter.keys())
    cn_unique = set(cn_counter.keys()) - set(hk_counter.keys())
    hk_unique = set(hk_counter.keys()) - set(cn_counter.keys())

    # 计算词频差异
    word_differences = {}
    for word in common_words:
        cn_freq = cn_counter[word]
        hk_freq = hk_counter[word]
        ratio = hk_freq / cn_freq if cn_freq > 0 else float("inf")
        word_differences[word] = {
            "cn_freq": cn_freq,
            "hk_freq": hk_freq,
            "ratio": ratio,
        }

    return {
        "cn_word_count": len(cn_filtered),
        "hk_word_count": len(hk_filtered),
        "common_words": len(common_words),
        "cn_unique_words": len(cn_unique),
        "hk_unique_words": len(hk_unique),
        "top_common_words": sorted(
            word_differences.items(),
            key=lambda x: x[1]["cn_freq"] + x[1]["hk_freq"],
            reverse=True,
        )[:20],
        "cn_unique_sample": list(cn_unique)[:10],
        "hk_unique_sample": list(hk_unique)[:10],
        "word_differences": word_differences,
    }


if __name__ == "__main__":
    result = word_cloud(slogan_data, domain="slogan")
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # 可视化词频云
    # 字体设置，确保中文正常显示；如simhei.ttf不存在，请下载或改用其他中文ttf字体路径
    try:
        font_path = "simhei.ttf"
        # 检查字体文件是否存在
        import os

        if not os.path.isfile(font_path):
            font_path = None
            print("警告：simhei.ttf 字体文件未找到，中文可能不会正常显示。请确保字体文件存在！")
    except Exception as e:
        font_path = None
        print("字体文件检查异常:", e)

    cn_counter = {
        word: info["cn_freq"] for word, info in result["word_differences"].items()
    }
    hk_counter = {
        word: info["hk_freq"] for word, info in result["word_differences"].items()
    }

    # 只可视化频率较高的词
    if font_path is not None:
        cn_cloud = WordCloud(
            font_path=font_path, width=600, height=400, background_color="white"
        ).generate_from_frequencies(cn_counter)
        hk_cloud = WordCloud(
            font_path=font_path, width=600, height=400, background_color="white"
        ).generate_from_frequencies(hk_counter)
    else:
        cn_cloud = WordCloud(
            width=600, height=400, background_color="white"
        ).generate_from_frequencies(cn_counter)
        hk_cloud = WordCloud(
            width=600, height=400, background_color="white"
        ).generate_from_frequencies(hk_counter)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cn_cloud, interpolation="bilinear")
    plt.title("CN词云", fontproperties="SimHei")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(hk_cloud, interpolation="bilinear")
    plt.title("HK词云", fontproperties="SimHei")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 画频次前20的中港词条对比条形图
    top_words = result["top_common_words"]
    top_cn = [w[0] for w in top_words]
    top_cn_freq = [w[1]["cn_freq"] for w in top_words]
    top_hk_freq = [w[1]["hk_freq"] for w in top_words]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(top_cn))
    width = 0.4
    plt.bar(x, top_cn_freq, width=width, label="CN", alpha=0.7)
    plt.bar(x + width, top_hk_freq, width=width, label="HK", alpha=0.7)
    # 使用SimHei字体显示中文标签
    plt.xticks(x + width / 2, top_cn, fontproperties="SimHei", rotation=45)
    plt.ylabel("频次", fontproperties="SimHei")
    plt.title("中港高频词条对比", fontproperties="SimHei")
    plt.legend(prop={"family": "SimHei"})
    plt.tight_layout()
    plt.show()

    print(
        "提示：如果图上没有显示中文内容，请确保已安装支持中文的字体（如SimHei.ttf），并检查 font_path 路径。可手动指定 WordCloud 的 font_path 参数。"
    )
