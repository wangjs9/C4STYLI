import json
import os
import random
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import font_manager
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  RANSACRegressor, TheilSenRegressor)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import norm_text

times_font = font_manager.FontProperties(fname="C4Styli/Times-New-Roman-Bold.ttf")
cambria_font = font_manager.FontProperties(fname="C4Styli/cambria-math.ttf")

MODEL_PATH = "/data/models/Qwen3-8B"
LAYER_IDX = -4
OUTPUT_DIR = "style_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


titles_cn_prompt = """英文标题：{title_en}
中文译名：{title}"""

titles_hk_prompt = """英文標題：{title_en}
中文譯名：{title}"""

slogans_cn_prompt = """品牌名称：{company}
{product_info}{date_info}广告标语: {slogan}"""

slogans_hk_prompt = """品牌名稱：{company}
{product_info}{date_info}廣告標語：{slogan}"""


def load_model():
    print(f"🚀 Loading model from {MODEL_PATH}...")
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen3-8B",
        hf_model=AutoModelForCausalLM.from_pretrained(MODEL_PATH),
        tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        move_to_device=True,
    )
    return model


def get_shuffled_text(text, tokenizer):
    """
    使用模型原生的 Tokenizer 进行词元化打乱，排除分词器不一致的干扰
    """
    if not text:
        return ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    random.shuffle(token_ids)
    shuffled_text = tokenizer.decode(token_ids)

    return shuffled_text


def get_activations(text_list, model, layer_idx):
    n_layers = model.cfg.n_layers
    real_idx = n_layers + layer_idx if layer_idx < 0 else layer_idx

    all_acts = []
    for text in tqdm(text_list, desc=f"Extrating Layer {real_idx}"):
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: f"blocks.{real_idx}.hook_resid_post" in n
            )
            # 提取输入文本最后一个 Token 的残差流
            act = cache[f"blocks.{real_idx}.hook_resid_post"][0, -1, :].cpu().numpy()
            all_acts.append(act)
    return np.array(all_acts)


def get_or_load_acts(texts, name, model, layer):
    path = os.path.join(OUTPUT_DIR, f"test_acts_{name}.npy")
    if os.path.exists(path):
        return np.load(path), model
    if model is None:
        model = load_model()
    acts = get_activations(texts, model, layer)
    np.save(path, acts)
    return acts, model


def calculate_gini(x):
    # 确保输入是排序好的正值
    if x.sum() == 0:
        return 0
    n = len(x)
    x = np.sort(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))


def compute_integrated_gradients(text, model, probe, scaler, pca, steps=30):
    # a. 获取输入 Embedding
    token_ids = model.to_tokens(text, prepend_bos=False)
    # 假设我们是在最后一层（LAYER_IDX）提取特征
    original_embeddings = model.embed(token_ids).detach()
    original_embeddings.requires_grad = True

    # b. 获取探针参数
    W = torch.tensor(probe.coef_[0], dtype=torch.float32).to(original_embeddings.device)
    # 先标准化，再经过PCA投影到探针空间，需要恢复为PCA空间的mean/scale
    mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(
        original_embeddings.device
    )
    scale = torch.tensor(scaler.scale_, dtype=torch.float32).to(
        original_embeddings.device
    )

    # PCA: transform (x - mean)/scale -> 提取pca.components_, pca.mean_
    pca_components = torch.tensor(pca.components_, dtype=torch.float32).to(
        original_embeddings.device
    )
    pca_mean = torch.tensor(pca.mean_, dtype=torch.float32).to(
        original_embeddings.device
    )

    # 用于将激活标准化并投影到PCA空间
    def transform_with_pca(act_vec):
        normed = (act_vec - mean) / scale
        pcaed = torch.matmul(normed - pca_mean, pca_components.T)
        return pcaed

    # c. 定义从 Embedding 到 Logit 的线性映射
    # 这里的简化逻辑是基于：线性探针的激活本质上是 Embedding 经过 Transformer 层后的投影
    # 如果你已经在特定的 LAYER_IDX 提取了激活，建议直接对激活做 IG

    # 我们这里以计算“特征向量对探针评分的贡献”为例
    # 获取特定层的激活 (Activation)
    with torch.no_grad():
        # 获取该文本在 LAYER_IDX 的残差流激活 [1, seq_len, d_model]
        act = get_activations([text], model, LAYER_IDX)
        act_vec = torch.tensor(act, dtype=torch.float32).squeeze(
            0
        )  # [seq_len, d_model]

    # 计算各 Token 的贡献：Score = (Act - Mean)/Scale * W
    # 这是线性探针特有的快捷方式：对于线性模型，IG 等价于 (输入 - 基准) * 权重
    # 假设基准是 0 向量
    importance_scores = ((act_vec - mean) / scale) * W
    token_importance = importance_scores.sum(dim=-1).abs().numpy()

    # 返回 Token 字符串及其对应分数
    tokens = model.to_str_tokens(text)
    return list(zip(tokens, token_importance))


def plot_style_violin(df, metric="Gini"):
    plt.figure(figsize=(8, 6))

    # 设置调色板
    palette = {"Mainland China": "#ff7f0e", "Hong Kong": "#1f77b4"}

    # 绘制小提琴图
    sns.violinplot(
        x="Region", y=metric, data=df, inner="quartile", palette=palette, alpha=0.7
    )

    # 叠加散点（让审稿人看到每一个样本点，增强透明度）
    sns.stripplot(
        x="Region", y=metric, data=df, color="black", size=3, jitter=True, alpha=0.3
    )

    # 标注标题与轴线
    title_map = {
        "Gini": "Distribution of Feature Sparsity (Gini Coefficient)",
        "Max_Contribution": "Influence of Top-1 Anchor Token",
    }
    plt.title(title_map.get(metric, metric), fontproperties=times_font, fontsize=14)
    plt.ylabel("Score", fontproperties=times_font)
    plt.xlabel("Region", fontproperties=times_font)

    # 计算显著性并标注 (T-test)
    cn_vals = df[df["Region"] == "Mainland China"][metric]
    hk_vals = df[df["Region"] == "Hong Kong"][metric]
    t_stat, p_val = stats.ttest_ind(cn_vals, hk_vals)

    # 在图上写 P 值
    plt.text(
        0.5,
        df[metric].max() * 0.95,
        f"p-value: {p_val:.2e}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="red",
        fontweight="bold",
    )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def collect_batch_statistics(texts_cn, texts_hk, model, probe, pca, scaler):
    data_records = []

    # 处理内地文本
    for text in tqdm(texts_cn, desc="Processing Mainland China"):
        attr = compute_integrated_gradients(text, model, probe, pca, scaler)
        scores = np.array([s for t, s in attr])
        norm_scores = scores / (scores.sum() + 1e-9)

        data_records.append(
            {
                "Region": "Mainland China",
                "Gini": calculate_gini(norm_scores),
                "Max_Contribution": np.max(norm_scores),  # 补充：最高贡献度词的影响力
            }
        )

    # 处理香港文本
    for text in tqdm(texts_hk, desc="Processing Hong Kong"):
        attr = compute_integrated_gradients(text, model, probe, pca, scaler)
        scores = np.array([s for t, s in attr])
        norm_scores = scores / (scores.sum() + 1e-9)

        data_records.append(
            {
                "Region": "Hong Kong",
                "Gini": calculate_gini(norm_scores),
                "Max_Contribution": np.max(norm_scores),
            }
        )

    return pd.DataFrame(data_records)


def run_full_attribution_analysis(texts_cn, texts_hk, model, probe, pca, scaler):
    # 1. 批量收集统计数据
    print("开始批量归因分析...")
    df_stats = collect_batch_statistics(texts_cn, texts_hk, model, probe, pca, scaler)

    # 2. 打印数值摘要
    print("\n--- 统计摘要 ---")
    print(df_stats.groupby("Region")["Gini"].describe())

    plot_style_violin(df_stats, metric="Gini")
    plot_style_violin(df_stats, metric="Max_Contribution")

    return df_stats


def filter_outliers_by_clustering(p_orig, p_shuf, eps=0.1, min_samples=5):
    """
    使用 DBSCAN 聚类算法识别并去除离群点
    eps: 两个样本被视为邻居的最大距离 (越小越严格，剔除的点越多)
    min_samples: 成为核心点所需的邻居数量
    """
    if len(p_orig) <= min_samples:
        return p_orig, p_shuf

    # 1. 数据准备与标准化 (聚类前必须缩放，使两个轴具有相同的权重)
    X = np.stack([p_orig, p_shuf], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 执行 DBSCAN
    # eps 决定了簇的“扩张”程度。如果点分布很密集，eps 可以设小一点
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_

    # labels == -1 的点是离群值
    mask = labels != -1

    # 打印剔除信息，方便调试
    n_outliers = np.sum(labels == -1)
    print(f"DBSCAN 识别并剔除离群点数量: {n_outliers} / {len(p_orig)}")

    return p_orig[mask], p_shuf[mask]


def draw_line(x_data, y_data, color, style, normalize=False):
    if normalize:
        X = x_data.reshape(-1, 1)
        y = y_data

        model = RANSACRegressor(random_state=1024)
        model.fit(X, y)

        x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
        y_plot = model.predict(x_plot)

        plt.plot(x_plot, y_plot, color=color, linestyle=style, linewidth=2.5)
    else:
        a, b = np.polyfit(x_data, y_data, 1)
        x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
        y_plot = a * x_plot + b
        plt.plot(x_plot, y_plot, color=color, linestyle=style, linewidth=2.5)


def plot_scatter_comparison(p_cn_orig, p_cn_shuf, p_hk_orig, p_hk_shuf, filename):
    # new_p_hk_orig, new_p_hk_shuf = [], []

    # for orig, shuf in zip(p_hk_orig, p_hk_shuf):
    #     if orig > 0.7 and shuf < 0.3:
    #         continue
    #     else:
    #         new_p_hk_orig.append(orig)
    #         new_p_hk_shuf.append(shuf)
    # p_hk_orig, p_hk_shuf = np.array(new_p_hk_orig), np.array(new_p_hk_shuf)
    # p_cn_orig, p_cn_shuf = filter_outliers_by_clustering(p_cn_orig, p_cn_shuf, eps=0.1)
    # p_hk_orig, p_hk_shuf = filter_outliers_by_clustering(p_hk_orig, p_hk_shuf, eps=0.1)

    plt.figure(figsize=(4, 4), dpi=300)

    # --- HK ---
    plt.scatter(
        p_hk_orig,
        p_hk_shuf,
        alpha=0.8,
        c="#1f77b4",
        edgecolors="white",
        label="HK",
        s=38,
        marker="^",
    )
    # draw_line(p_hk_orig, p_hk_shuf, "#1f77b4", "--", "HK")

    # --- CN ---
    plt.scatter(
        p_cn_orig,
        p_cn_shuf,
        alpha=0.8,
        c="#ff7f0e",
        edgecolors="white",
        label="CN",
        s=28,
    )
    # draw_line(p_cn_orig, p_cn_shuf, "#ff7f0e", "--", "CN")

    plt.plot(
        [0, 1],
        [0, 1],
        color="#333333",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        zorder=2,
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xticks(fontproperties=cambria_font)
    plt.yticks(fontproperties=cambria_font)

    plt.xlabel("Original Sequence", fontproperties=times_font, fontsize=10)
    plt.ylabel("Sequence with Shuffled Tokens", fontproperties=times_font, fontsize=10)

    plt.title(
        "Stylistic Probability under Structural Ablation",
        fontproperties=times_font,
        fontsize=10,
    )

    plt.grid(True, linestyle=":", alpha=0.6)

    leg = plt.legend(
        prop=cambria_font,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
        facecolor="#F5F5F5",
        edgecolor="none",
    )
    leg.get_frame().set_alpha(0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight", pad_inches=0
    )


def remove_outliers_zscore(df, threshold=3):
    # 只针对需要筛选的列计算
    z_scores = np.abs(stats.zscore(df[["Orig", "Shuf"]]))
    # 只要 Orig 或 Shuf 有一个超出阈值就剔除
    filtered_entries = (z_scores < threshold).all(axis=1)
    return df[filtered_entries]


def plot_density_comparison(p_cn_orig, p_cn_shuf, p_hk_orig, p_hk_shuf, filename):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # 1. 绘制背景色块填充 (2D Histogram)
    # 使用较小的 bins 来创造“块状感”，alpha 控制透明度
    ax.hist2d(p_cn_orig, p_cn_shuf, bins=25, cmap="Oranges", alpha=0.3, zorder=1)
    ax.hist2d(p_hk_orig, p_hk_shuf, bins=25, cmap="Blues", alpha=0.3, zorder=1)

    # 2. 叠加保留所有的点 (Scatter)
    # s=25, alpha=0.7, edgecolors='white', linewidth=0.4 是关键参数，让点有立体感且不完全覆盖背景
    ax.scatter(
        p_cn_orig,
        p_cn_shuf,
        color="#ff7f0e",
        s=25,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.4,
        label="CN",
        zorder=3,
    )

    ax.scatter(
        p_hk_orig,
        p_hk_shuf,
        color="#1f77b4",
        s=25,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.4,
        label="HK",
        zorder=4,
    )

    # 3. 绘制线性拟合趋势线
    # CN 趋势线
    if len(p_cn_orig) > 1:
        reg_cn = LinearRegression()
        reg_cn.fit(p_cn_orig.reshape(-1, 1), p_cn_shuf)
        x_fit_cn = np.array([p_cn_orig.min(), p_cn_orig.max()])  # 根据数据范围绘制
        ax.plot(
            x_fit_cn,
            reg_cn.predict(x_fit_cn.reshape(-1, 1)),
            color="#ff7f0e",
            linestyle="-",
            linewidth=2.5,
            alpha=0.9,
            zorder=6,
        )

    # HK 趋势线
    if len(p_hk_orig) > 1:
        reg_hk = LinearRegression()
        reg_hk.fit(p_hk_orig.reshape(-1, 1), p_hk_shuf)
        x_fit_hk = np.array([p_hk_orig.min(), p_hk_orig.max()])  # 根据数据范围绘制
        ax.plot(
            x_fit_hk,
            reg_hk.predict(x_fit_hk.reshape(-1, 1)),
            color="#1f77b4",
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            zorder=7,
        )

    # 4. 绘制对角参考线 (y=x)
    ax.plot(
        [0, 1],
        [0, 1],
        color="#333333",
        linestyle=":",
        linewidth=1.0,
        alpha=0.6,
        zorder=2,
    )  # 放在趋势线下面，但散点上面，不影响趋势线可见性

    # 5. 坐标轴与美化
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    # 字体设置 (沿用你之前的 Times New Roman 和 Cambria)
    ax.set_xlabel(
        "Original Sequence Confidence", fontproperties=times_font, fontsize=11
    )
    ax.set_ylabel("Shuffled Tokens Confidence", fontproperties=times_font, fontsize=11)
    ax.set_title(
        "Stylistic Scatter Plot under Structural Ablation",
        fontproperties=times_font,
        fontsize=12,
        pad=10,
    )

    # 刻度字体调整
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(cambria_font)

    # 6. 图例 (模仿示例图样式)
    # 这里手动创建 Line2D 对象来控制图例中线的样式，确保与图中拟合线一致
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="CN",
            markerfacecolor="#ff7f0e",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="HK",
            markerfacecolor="#1f77b4",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.4,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        facecolor="#f5f5f5",
        edgecolor="none",
        prop=cambria_font,
    )

    # 增加网格线
    ax.grid(True, linestyle=":", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    plt.show()


def load_training_hidden_states(model, layer_idx, region="hk"):
    if os.path.exists(os.path.join(OUTPUT_DIR, f"training_acts_{region}.npy")):
        return np.load(os.path.join(OUTPUT_DIR, f"training_acts_{region}.npy")), model
    else:
        if model is None:
            model = load_model()
        with open("C4Styli/titles/finetune_movie_titles.json", "r") as f:
            raw_data = json.load(f)
        texts = [
            titles_cn_prompt.format(
                title_en=item["TITLE"], title=item[f"TITLE ({region.upper()})"]
            )
            for item in raw_data
        ]

        with open("C4Styli/slogans/finetune_advertise_slogans.json", "r") as f:
            raw_data = json.load(f)
        texts += [
            slogans_cn_prompt.format(
                company=item["company"],
                product_info=item.get("product", "\n"),
                date_info=item["date"] if item["date"] != -1 else "\n",
                slogan=item["slogan"],
            )
            for item in raw_data
            if item["region"] == region.upper()
        ]

        texts = [norm_text(t, "HK") for t in texts] + [
            norm_text(t, "CN") for t in texts
        ]
        acts = get_activations(texts, model, layer_idx)
        np.save(os.path.join(OUTPUT_DIR, f"training_acts_{region}.npy"), acts)
        return acts, model


if __name__ == "__main__":
    confidence_comparison = False
    model = None

    acts_hk, model = load_training_hidden_states(model, LAYER_IDX, "hk")
    acts_cn, model = load_training_hidden_states(model, LAYER_IDX, "cn")

    X = np.vstack([acts_cn, acts_hk])
    y = np.array([0] * len(acts_cn) + [1] * len(acts_hk))

    if os.path.exists(os.path.join(OUTPUT_DIR, "probe.joblib")):
        probe_data = joblib.load(os.path.join(OUTPUT_DIR, "probe.joblib"))
        scaler = probe_data["scaler"]
        pca = pca = PCA(n_components=0.8, random_state=42)
        probe = probe_data["probe"]
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pca = PCA(n_components=0.8, random_state=42)
        X = pca.fit_transform(X)
        probe = LogisticRegression(
            max_iter=300, penalty="l2", solver="liblinear", random_state=1
        )
        # probe = LinearRegression()
        # probe = TheilSenRegressor(random_state=42)
        probe.fit(X, y)
        joblib.dump(
            {"scaler": scaler, "probe": probe, "pca": pca},
            os.path.join(OUTPUT_DIR, "probe.joblib"),
        )

    with open("C4Styli/titles/val_movie_titles.json", "r") as f:
        raw_data = json.load(f)
    test_cn_titles = [item["TITLE (CN)"] for item in raw_data]
    cn_titles_data = [item for item in raw_data]
    test_hk_titles = [item["TITLE (HK)"] for item in raw_data]
    hk_titles_data = [item for item in raw_data]
    with open("C4Styli/slogans/val_advertise_slogans.json", "r") as f:
        raw_data = json.load(f)
    test_cn_slogans = [item["slogan"] for item in raw_data if item["region"] == "CN"]
    cn_slogans_data = [item for item in raw_data if item["region"] == "CN"]
    test_hk_slogans = [item["slogan"] for item in raw_data if item["region"] == "HK"]
    hk_slogans_data = [item for item in raw_data if item["region"] == "HK"]

    if confidence_comparison:
        if not os.path.exists(os.path.join(OUTPUT_DIR, "shuffled_test_data.json")):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            shuffled_test_cn_titles = [
                get_shuffled_text(t, tokenizer) for t in test_cn_titles
            ]
            shuffled_test_cn_slogans = [
                get_shuffled_text(t, tokenizer) for t in test_cn_slogans
            ]
            shuffled_test_hk_titles = [
                get_shuffled_text(t, tokenizer) for t in test_hk_titles
            ]
            shuffled_test_hk_slogans = [
                get_shuffled_text(t, tokenizer) for t in test_hk_slogans
            ]
            json.dump(
                {
                    "shuffled_test_cn_titles": shuffled_test_cn_titles,
                    "shuffled_test_cn_slogans": shuffled_test_cn_slogans,
                    "shuffled_test_hk_titles": shuffled_test_hk_titles,
                    "shuffled_test_hk_slogans": shuffled_test_hk_slogans,
                },
                open(
                    os.path.join(OUTPUT_DIR, "shuffled_test_data.json"),
                    "w",
                    encoding="utf-8",
                ),
                ensure_ascii=False,
                indent=4,
            )
        else:
            test_data = json.load(
                open(
                    os.path.join(OUTPUT_DIR, "shuffled_test_data.json"),
                    "r",
                    encoding="utf-8",
                )
            )
            shuffled_test_cn_titles = test_data["shuffled_test_cn_titles"]
            shuffled_test_cn_slogans = test_data["shuffled_test_cn_slogans"]
            shuffled_test_hk_titles = test_data["shuffled_test_hk_titles"]
            shuffled_test_hk_slogans = test_data["shuffled_test_hk_slogans"]

        assert (
            len(cn_titles_data) == len(test_cn_titles) == len(shuffled_test_cn_titles)
        )
        assert (
            len(hk_titles_data) == len(test_hk_titles) == len(shuffled_test_hk_titles)
        )
        assert (
            len(cn_slogans_data)
            == len(test_cn_slogans)
            == len(shuffled_test_cn_slogans)
        )
        assert (
            len(hk_slogans_data)
            == len(test_hk_slogans)
            == len(shuffled_test_hk_slogans)
        )

        # --- 3. 概率预测与对比分析 ---
        def get_probs(acts, target_class):
            acts = scaler.transform(acts)
            acts = pca.transform(acts)
            try:
                return probe.predict_proba(acts)[:, target_class]
            except:
                return probe.predict(acts)

        if not os.path.exists(os.path.join(OUTPUT_DIR, "probability_scores.json")):
            test_cn_texts = [
                titles_cn_prompt.format(title_en=item["TITLE"], title=test_cn_titles[i])
                for i, item in enumerate(cn_titles_data)
            ] + [
                slogans_cn_prompt.format(
                    company=item["company"],
                    product_info=item.get("product", "\n"),
                    date_info=item["date"] if item["date"] != -1 else "\n",
                    slogan=test_cn_slogans[i],
                )
                for i, item in enumerate(cn_slogans_data)
            ]
            test_cn_texts = [norm_text(t, "CN") for t in test_cn_texts]
            test_hk_texts = [
                titles_cn_prompt.format(title_en=item["TITLE"], title=test_hk_titles[i])
                for i, item in enumerate(hk_titles_data)
            ] + [
                slogans_cn_prompt.format(
                    company=item["company"],
                    product_info=item.get("product", "\n"),
                    date_info=item["date"] if item["date"] != -1 else "\n",
                    slogan=test_hk_slogans[i],
                )
                for i, item in enumerate(hk_slogans_data)
            ]
            test_hk_texts = [norm_text(t, "CN") for t in test_hk_texts]
            shuffled_test_cn_texts = [
                titles_cn_prompt.format(
                    title_en=item["TITLE"], title=shuffled_test_cn_titles[i]
                )
                for i, item in enumerate(cn_titles_data)
            ] + [
                slogans_cn_prompt.format(
                    company=item["company"],
                    product_info=item.get("product", "\n"),
                    date_info=item["date"] if item["date"] != -1 else "\n",
                    slogan=shuffled_test_cn_slogans[i],
                )
                for i, item in enumerate(cn_slogans_data)
            ]
            shuffled_test_cn_texts = [
                norm_text(t, "CN") for t in shuffled_test_cn_texts
            ]
            shuffled_test_hk_texts = [
                titles_cn_prompt.format(
                    title_en=item["TITLE"], title=shuffled_test_hk_titles[i]
                )
                for i, item in enumerate(hk_titles_data)
            ] + [
                slogans_cn_prompt.format(
                    company=item["company"],
                    product_info=item.get("product", "\n"),
                    date_info=item["date"] if item["date"] != -1 else "\n",
                    slogan=shuffled_test_hk_slogans[i],
                )
                for i, item in enumerate(hk_slogans_data)
            ]
            shuffled_test_hk_texts = [
                norm_text(t, "CN") for t in shuffled_test_hk_texts
            ]

            test_acts_cn_orig, model = get_or_load_acts(
                test_cn_texts, "cn_orig", model, LAYER_IDX
            )
            test_acts_hk_orig, model = get_or_load_acts(
                test_hk_texts, "hk_orig", model, LAYER_IDX
            )
            test_acts_cn_shuf, model = get_or_load_acts(
                shuffled_test_cn_texts, "cn_shuf", model, LAYER_IDX
            )
            test_acts_hk_shuf, model = get_or_load_acts(
                shuffled_test_hk_texts, "hk_shuf", model, LAYER_IDX
            )
            p_cn_orig = get_probs(test_acts_cn_orig, 0)
            p_cn_shuf = get_probs(test_acts_cn_shuf, 0)
            p_hk_orig = get_probs(test_acts_hk_orig, 1)
            p_hk_shuf = get_probs(test_acts_hk_shuf, 1)
            json.dump(
                {
                    "p_cn_orig": p_cn_orig.tolist(),
                    "p_cn_shuf": p_cn_shuf.tolist(),
                    "p_hk_orig": p_hk_orig.tolist(),
                    "p_hk_shuf": p_hk_shuf.tolist(),
                },
                open(
                    os.path.join(OUTPUT_DIR, "probability_scores.json"),
                    "w",
                    encoding="utf-8",
                ),
                ensure_ascii=False,
                indent=4,
            )
        else:
            probability_scores = json.load(
                open(
                    os.path.join(OUTPUT_DIR, "probability_scores.json"),
                    "r",
                    encoding="utf-8",
                )
            )
            p_cn_orig = np.array(probability_scores["p_cn_orig"])
            p_cn_shuf = np.array(probability_scores["p_cn_shuf"])
            p_hk_orig = np.array(probability_scores["p_hk_orig"])
            p_hk_shuf = np.array(probability_scores["p_hk_shuf"])

        # --- 4. 核心指标打印 ---
        print("\n" + "=" * 50)
        print("📊 跨数据集结构消融实验结果 (Generalization Test)")
        print("=" * 50)
        print(f"CN 测试集平均概率 (有序): {p_cn_orig.mean():.4f}")
        print(f"CN 测试集平均概率 (乱序): {p_cn_shuf.mean():.4f}")
        print(f"HK 测试集平均概率 (有序): {p_hk_orig.mean():.4f}")
        print(f"HK 测试集平均概率 (乱序): {p_hk_shuf.mean():.4f}")

        plot_scatter_comparison(
            p_cn_orig, p_cn_shuf, p_hk_orig, p_hk_shuf, "confidence_comparison.pdf"
        )

        plot_density_comparison(
            p_cn_orig, p_cn_shuf, p_hk_orig, p_hk_shuf, "density_heatmap.pdf"
        )

    else:
        if model is None:
            model = load_model()
        test_cn_texts = [
            titles_cn_prompt.format(title_en=item["TITLE"], title=test_cn_titles[i])
            for i, item in enumerate(cn_titles_data)
        ] + [
            slogans_cn_prompt.format(
                company=item["company"],
                product_info=item.get("product", "\n"),
                date_info=item["date"] if item["date"] != -1 else "\n",
                slogan=test_cn_slogans[i],
            )
            for i, item in enumerate(cn_slogans_data)
        ]
        test_hk_texts = [
            titles_cn_prompt.format(title_en=item["TITLE"], title=test_hk_titles[i])
            for i, item in enumerate(hk_titles_data)
        ] + [
            slogans_cn_prompt.format(
                company=item["company"],
                product_info=item.get("product", "\n"),
                date_info=item["date"] if item["date"] != -1 else "\n",
                slogan=test_hk_slogans[i],
            )
            for i, item in enumerate(hk_slogans_data)
        ]
        test_cn_texts = test_cn_texts[:50]
        test_hk_texts = test_hk_texts[:50]
        results_df = run_full_attribution_analysis(
            test_cn_texts[:50], test_hk_texts[:50], model, probe, pca, scaler
        )
