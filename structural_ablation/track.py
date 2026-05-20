import torch
import json
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from tqdm import tqdm

times_font = font_manager.FontProperties(fname='C4Styli/Times-New-Roman-Bold.ttf')
cambria_font = font_manager.FontProperties(fname='C4Styli/cambria-math.ttf')

MODEL_PATH = "/data/models/Qwen3-8B"
LAYER_IDX = -2
OUTPUT_DIR = "style_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib.pyplot as plt
import platform

# 1. 自动搜索并设置中文字体
def set_chinese_font():
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    
    # 尝试设置常用中文字体
    system = platform.system()
    if system == "Linux":
        # 很多服务器自带这个字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    elif system == "Darwin":  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    elif system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei']
    
    print(f"当前使用的字体: {plt.rcParams['font.sans-serif']}")

set_chinese_font()

# ==========================================
# 2. 模型加载与初始化
# ==========================================
def load_model():
    print(f"🚀 Loading model from {MODEL_PATH}...")
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen3-8B",
        hf_model=AutoModelForCausalLM.from_pretrained(MODEL_PATH),
        tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH), 
        device="cuda:7" if torch.cuda.is_available() else "cpu",
        move_to_device=True
    )
    return model

# ==========================================
# 3. 激活值提取 (核心诊断逻辑)
# ==========================================
def get_activations(text_list, model, layer_idx):
    n_layers = model.cfg.n_layers
    real_idx = n_layers + layer_idx if layer_idx < 0 else layer_idx
    
    all_acts = []
    for text in tqdm(text_list, desc=f"Extrating Layer {real_idx}"):
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: f"blocks.{real_idx}.hook_resid_post" in n
            )
            # 提取输入文本最后一个 Token 的残差流
            act = cache[f"blocks.{real_idx}.hook_resid_post"][0, -1, :].cpu().numpy()
            all_acts.append(act)
    return np.array(all_acts)

# ==========================================
# 4. 肤浅性分析：寻找 Top 贡献词汇
# ==========================================
def analyze_token_bias(model, text_list, style_vector, scaler, layer_idx, top_k=30):
    """
    分析全语料中对风格维度贡献最大的词，用于证明其肤浅性
    """
    n_layers = model.cfg.n_layers
    real_idx = n_layers + layer_idx if layer_idx < 0 else layer_idx
    results = []
    
    for text in tqdm(text_list[:200], desc="Analyzing Token Saliency"): # 采样分析
        tokens = model.to_str_tokens(text)
        token_ids = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(token_ids, names_filter=lambda n: f"blocks.{real_idx}.hook_resid_post" in n)
            resids = cache[f"blocks.{real_idx}.hook_resid_post"][0]
        
        # 投影到风格轴
        resids_scaled = scaler.transform(resids.cpu().numpy())
        scores = np.dot(resids_scaled, style_vector)
        
        for t, s in zip(tokens, scores):
            results.append({"token": t, "score": s})

    df = pd.DataFrame(results)
    # 统计 HK 偏向最强的词和 CN 偏向最强的词
    top_hk = df.groupby('token')['score'].mean().sort_values(ascending=False).head(top_k)
    top_cn = df.groupby('token')['score'].mean().sort_values(ascending=True).head(top_k)
    return top_hk, top_cn

def plot_neurons(indices, importance, output_path):
    plt.figure(figsize=(10, 8))
    vals = importance[indices]
    labels = [f"N{i}" for i in indices]
    colors = ['#1f77b4' if v > 0 else '#ff7f0e' for v in vals]
    sns.barplot(x=vals, y=labels, palette=colors)
    plt.title("Top 20 Stylistic Neurons (Neuron-level Analysis)", fontproperties=times_font, fontsize=14)
    plt.savefig(output_path)
    plt.close()


def get_token_diagnostics(model, text_list, style_vector, scaler, layer_idx):
    """
    看看模型在判断风格时，到底在看哪些词
    """
    n_layers = model.cfg.n_layers
    real_idx = n_layers + layer_idx if layer_idx < 0 else layer_idx
    token_data = []
    
    # 遍历文本提取每个 Token 的投影得分
    for text in text_list:
        tokens = model.to_str_tokens(text)
        token_ids = model.to_tokens(text)
        
        with torch.no_grad():
            # 1. 运行模型并获取缓存
            _, cache = model.run_with_cache(
                token_ids, 
                names_filter=lambda n: f"blocks.{real_idx}.hook_resid_post" in n
            )
            # 2. 正确赋值 resids (从 cache 中提取)
            resids = cache[f"blocks.{real_idx}.hook_resid_post"][0] # [seq_len, d_model]
        
        # 3. 标准化激活值
        scaled_acts = scaler.transform(resids.cpu().numpy())
        scores = np.dot(scaled_acts, style_vector)
        
        for t, s in zip(tokens, scores):
            token_data.append({"token": t, "score": s})
    
    df = pd.DataFrame(token_data)
    
    # --- 关键修正：分别排序 ---
    
    # 1. 寻找最能代表 HK 的词：得分越高（越正），HK 属性越强
    # 这些词如果是“嘅”、“咗”，证明 HK 判别是肤浅的
    top_hk_tokens = df.groupby('token')['score'].mean().sort_values(ascending=False).head(20)
    
    # 2. 寻找最能代表 CN 的词：得分越低（越负），CN 属性越强
    # 这些词如果是“的”、“了”或者特定简体词，证明 CN 判别也是肤浅的
    top_cn_tokens = df.groupby('token')['score'].mean().sort_values(ascending=True).head(20)
    
    return top_hk_tokens, top_cn_tokens

def plot_token_ranking(hk_t, cn_t, output_path):
    plt.figure(figsize=(12, 8))
    combined = pd.concat([hk_t, cn_t])
    colors = ['#1f77b4']*20 + ['#ff7f0e']*20
    combined.plot(kind='barh', color=colors)
    plt.title("Top Tokens Driving the 'Style Neurons' (Token-level Evidence)", fontproperties=times_font, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

import jieba
import random
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

def batch_structural_ablation_jieba(model, text_list, style_vector, scaler, layer_idx):
    """
    使用 jieba 分词打乱语序，测试得分保留率
    """
    results = []
    n_layers = model.cfg.n_layers
    real_idx = n_layers + layer_idx if layer_idx < 0 else layer_idx

    def get_style_score(t):
        if not t.strip(): return 0
        tokens = model.to_tokens(t)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, 
                names_filter=lambda n: f"blocks.{real_idx}.hook_resid_post" in n
            )
            act = cache[f"blocks.{real_idx}.hook_resid_post"][0, -1, :].cpu().numpy()
        
        act_scaled = scaler.transform(act.reshape(1, -1))
        return np.dot(act_scaled, style_vector)[0]

    for orig_text in tqdm(text_list[:300], desc="Jieba 词级消融测试"):
        # 1. 原始得分
        score_orig = get_style_score(orig_text)
        
        # 2. 使用 jieba 分词并随机打乱
        seg_list = list(jieba.cut(orig_text))
        random.shuffle(seg_list)
        shuffled_text = "".join(seg_list)
        
        # 3. 计算打乱后的得分
        score_shuffled = get_style_score(shuffled_text)
        
        results.append({
            "original": orig_text,
            "shuffled": shuffled_text,
            "score_orig": score_orig,
            "score_shuffled": score_shuffled,
            # 计算绝对值的保留率，避免正负抵消
            "retention": abs(score_shuffled / score_orig) if score_orig != 0 else 0
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    model = load_model()
    
    with open("C4Styli/titles/finetune_movie_titles.json", "r") as f:
        raw_data = json.load(f)
    cn_texts = [item["TITLE (CN)"] for item in raw_data]
    hk_texts = [item["TITLE (HK)"] for item in raw_data]
    with open("C4Styli/slogans/finetune_advertise_slogans.json", "r") as f:
        raw_data = json.load(f)
    cn_texts += [item["slogan"] for item in raw_data if item["region"] == "CN"]
    hk_texts += [item["slogan"] for item in raw_data if item["region"] == "HK"]
    
    if os.path.exists(os.path.join(OUTPUT_DIR, "acts_cn.npy")):
        acts_cn = np.load(os.path.join(OUTPUT_DIR, "acts_cn.npy"))
    else:
        acts_cn = get_activations(cn_texts, model, LAYER_IDX)
        np.save(os.path.join(OUTPUT_DIR, "acts_cn.npy"), acts_cn)
        
    if  os.path.exists(os.path.join(OUTPUT_DIR, "acts_hk.npy")):
        acts_hk = np.load(os.path.join(OUTPUT_DIR, "acts_hk.npy"))
    else:
        acts_hk = get_activations(hk_texts, model, LAYER_IDX)
        np.save(os.path.join(OUTPUT_DIR, "acts_hk.npy"), acts_hk)
    
    
    X = np.vstack([acts_cn, acts_hk])
    y = np.array([0]*len(acts_cn) + [1]*len(acts_hk))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    probe = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    probe.fit(X_scaled, y)
    
    style_importance = probe.coef_[0]
    
    # 诊断分析：获取 Top 贡献词
    print("\n🧐 Analyzing if the model is superficial...")
    hk_bias_tokens, cn_bias_tokens = analyze_token_bias(model, cn_texts + hk_texts, style_importance, scaler, LAYER_IDX)
    
    print("\n--- Top HK-biased Tokens (Evidence of Dialect/Pattern) ---")
    print(hk_bias_tokens)
    print("\n--- Top CN-biased Tokens (Evidence of Dialect/Pattern) ---")
    print(cn_bias_tokens)
    
    # 保存结果
    joblib.dump({'vector': style_importance, 'scaler': scaler, 'hk_tokens': hk_bias_tokens}, 
                f"{OUTPUT_DIR}/diagnostic_results.pkl")
    
    abs_weights = np.abs(style_importance)
    top_neuron_indices = np.argsort(abs_weights)[-20:]
    top_neuron_indices = top_neuron_indices[np.argsort(abs_weights[top_neuron_indices])[::-1]]
    
        # --- 运行实验 ---
    df_jieba = batch_structural_ablation_jieba(model, cn_texts + hk_texts, style_importance, scaler, LAYER_IDX)

    # 统计分析
    avg_retention = df_jieba['retention'].mean()
    print(f"\n📊 Jieba 分词打乱实验结果:")
    print(f"平均得分保留率 (Mean Retention): {avg_retention:.2%}")

    # 展示几个典型例子
    print("\n📝 典型案例对比:")
    print(df_jieba[['original', 'shuffled', 'retention']].head(10))
    
    plot_neurons(top_neuron_indices, style_importance, os.path.join(OUTPUT_DIR, "neuron_importance.png"))
    # hk_tokens, cn_tokens = get_token_diagnostics(model, cn_texts + hk_texts, style_importance, scaler, LAYER_IDX)
    # plot_token_ranking(hk_tokens, cn_tokens, os.path.join(OUTPUT_DIR, "token_evidence.png"))