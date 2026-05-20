
import json
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jieba.posseg as pseg
import re
from utils import norm_text
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager
from transformers import MarianMTModel, MarianTokenizer
import torch
from pysenti import ModelClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
senti = ModelClassifier()
vader_analyzer = SentimentIntensityAnalyzer()

def translate_zh_to_en(text, tokenizer, model) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        translated = model.generate(
            **inputs,
            max_length=64,
            num_beams=5,
            early_stopping=True
        )
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return 1.0 - np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    )


def year_to_bin(year: int, bin_size: int = 5, max_year: int = 2025, min_year: int = 1900) -> str:
    """
    Map a year to a coarse temporal bin.
    Example: 1993 -> 1990-1994
             2001 -> 2000-2004
    """
    start = (year // bin_size) * bin_size
    end = start + bin_size - 1
    if end > max_year:
        end = max_year
    if start < min_year:
        start = min_year
    return f"{start}-{end}"


def translation_distance(titles):
    # use API to translate the Chinese into English and compute the distance
    if os.path.exists("C4Styli/translation_movie_titles.json"):
        data = json.load(open("C4Styli/translation_movie_titles.json", "r", encoding="utf-8"))
        all_results = data["all"]
        year_results = data["year"]
    else:
        year_range = [t["year"] for t in titles if t["year"] != -1]
        max_year, min_year = max(year_range), min(year_range)
        MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"
        tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        model = MarianMTModel.from_pretrained(MODEL_NAME)
        model.eval()
        
        # Initialize sentence transformer for embeddings
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        all_results = {"CN": [], "HK": []}
        year_results = {"CN": defaultdict(list), "HK": defaultdict(list)}
        
        for item in tqdm(titles, total=len(titles), desc="Processing Titles"):
            title_cn = item["title_cn"]
            cn_lit = translate_zh_to_en(title_cn, tokenizer, model)
            title_hk = norm_text(item["title_hk"], "CN")
            hk_lit = translate_zh_to_en(title_hk, tokenizer, model)
            
            title_en = item["title"]
            
            # embed
            emb_en = embed_model.encode(title_en, normalize_embeddings=True)
            emb_cn = embed_model.encode(cn_lit, normalize_embeddings=True)
            emb_hk = embed_model.encode(hk_lit, normalize_embeddings=True)
            
            dist_cn = cosine_distance(emb_cn, emb_en)
            dist_hk = cosine_distance(emb_hk, emb_en)
            all_results["CN"].append(dist_cn)
            all_results["HK"].append(dist_hk)
            
            year = item["year"]
            if year != -1:
                y_bin = year_to_bin(year, 5, max_year, min_year)
                year_results["CN"][y_bin].append(dist_cn)
                year_results["HK"][y_bin].append(dist_hk)
                
        with open("C4Styli/translation_movie_titles.json", "w", encoding="utf-8") as f:
            json.dump({"all": all_results, "year": year_results}, f, ensure_ascii=False, indent=4)
    
    print(f"HK lit. distance {np.mean(all_results['HK'])}")
    print(f"CN lit. distance {np.mean(all_results['CN'])}")
    
    cn_means = {}
    hk_means = {}
    
    for year_bin, distances in year_results["CN"].items():
        cn_means[year_bin] = np.mean(distances)
    
    for year_bin, distances in year_results["HK"].items():
        hk_means[year_bin] = np.mean(distances)
    
    all_bins = sorted(set(cn_means.keys()) | set(hk_means.keys()))
    
    cn_values = [cn_means.get(bin, 0) for bin in all_bins]
    hk_values = [hk_means.get(bin, 0) for bin in all_bins]
    
    y_label = "lit-orig Distance"
    figure_name = "Interpretive Shift by Time Period"
    save_path = "C4Styli/translation_distance_by_year.pdf"
    visualization(cn_values, hk_values, all_bins, y_label, figure_name, save_path)
    
    print(f"图表已保存到: C4Styli/translation_distance_by_year.pdf")
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"CN - 时间周期数量: {len(cn_means)}")
    print(f"HK - 时间周期数量: {len(hk_means)}")
    print(f"CN - 平均距离范围: {min(cn_values):.3f} - {max(cn_values):.3f}")
    print(f"HK - 平均距离范围: {min(hk_values):.3f} - {max(hk_values):.3f}")


def english_ratio(data: list[dict], domain="Movie Titles"):
    """
    Compute the ratio of English content in movie titles over different years and create visualizations.
    Returns the overall English ratio across all data.
    """
    if os.path.exists(f"C4Styli/english_ratio_{domain.lower().replace(' ', '_')}.json"):
        cached_data = json.load(open(f"C4Styli/english_ratio_{domain.lower().replace(' ', '_')}.json", "r", encoding="utf-8"))
        all_results = cached_data["all"]
        year_results = cached_data["year"]
    else:
        year_range = [t["year"] for t in data if t["year"]!=-1]
        max_year, min_year = max(year_range), min(year_range)
        all_results = {"CN": [], "HK": []}
        year_results = {"CN": defaultdict(list), "HK": defaultdict(list)}

        for item in tqdm(data, total=len(data), desc="Processing English Ratios"):
            if domain == "Movie Titles":
                cn_ratio = compute_english_ratio(item["title_cn"])
                hk_ratio = compute_english_ratio(norm_text(item["title_hk"], "CN"))
                all_results["CN"].append(cn_ratio)
                all_results["HK"].append(hk_ratio)
                year = item.get("year", -1)
                if year != -1:
                    y_bin = year_to_bin(year, 5, max_year, min_year)
                    year_results["CN"][y_bin].append(cn_ratio)
                    year_results["HK"][y_bin].append(hk_ratio)
            elif domain == "Advertising Slogans":
                ratio = compute_english_ratio(norm_text(item["slogan"], "CN"))
                all_results[item["region"]].append(ratio)
                year = item.get("year", -1)
                if year != -1:
                    y_bin = year_to_bin(year, 5, max_year, min_year)
                    year_results[item["region"]][y_bin].append(ratio)
            else:
                raise ValueError(f"Invalid domain: {domain}")

        # Cache results
        with open(f"C4Styli/english_ratio_{domain.lower().replace(' ', '_')}.json", "w", encoding="utf-8") as f:
            json.dump({"all": all_results, "year": year_results}, f, ensure_ascii=False, indent=4)

    print(f"HK English ratio: {np.mean(all_results['HK']):.4f}")
    print(f"CN English ratio: {np.mean(all_results['CN']):.4f}")
    
    cn_means = {}
    hk_means = {}

    for year_bin, ratios in year_results["CN"].items():
        cn_means[year_bin] = np.mean(ratios)

    for year_bin, ratios in year_results["HK"].items():
        hk_means[year_bin] = np.mean(ratios)
    
    all_bins = sorted(set(cn_means.keys()) | set(hk_means.keys()))

    cn_values = [cn_means.get(bin, 0) for bin in all_bins]
    hk_values = [hk_means.get(bin, 0) for bin in all_bins]
        
        
    y_label = "English Ratio"
    figure_name = f"English Prevalence by Time Period"
    save_path = f"C4Styli/english_ratio_by_year_{domain}.pdf"
    if domain == "Movie Titles":
        figure_width = 6.5
        color_cn = '#5281C3'
        color_hk = '#BE514D'
    else:
        figure_width = 6
        color_cn = '#F8C452'
        color_hk = '#95C831'
    visualization(cn_values, hk_values, all_bins, y_label, figure_name, save_path, figure_width, color_cn, color_hk)
    
    print("\n统计信息:")
    print(f"CN - 时间周期数量: {len(cn_means)}")
    print(f"HK - 时间周期数量: {len(hk_means)}")
    print(f"CN - 平均英语比率范围: {min(cn_values):.4f} - {max(cn_values):.4f}")
    print(f"HK - 平均英语比率范围: {min(hk_values):.4f} - {max(hk_values):.4f}")


def compute_english_ratio(text: str) -> float:
    """
    Compute the ratio of English characters/words in a text.
    Returns a float between 0 and 1 representing the proportion of English content.
    """
    if not text:
        return 0.0
    
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    total_words = len(re.findall(r'\b\w+\b', text))

    return len(english_words) / total_words if total_words > 0 else 0.0


def emotion_density(text: str) -> float:
    res = senti.classify(text)
    return max(res["positive_prob"], res["negative_prob"])


def extract_emotion_words_and_score(text: str) -> dict:
    """
    Extract emotion words and their scores from English text using VADER.
    Returns a dictionary with emotion words, their individual scores, and overall sentiment.
    """
    if not text or not isinstance(text, str):
        return {"emotion_words": [], "word_scores": [], "overall_score": 0.0, "sentiment": "neutral"}

    # Use VADER for sentiment analysis
    vader_scores = vader_analyzer.polarity_scores(text)
    compound_score = vader_scores['compound']  # Overall sentiment score (-1 to 1)

    # Determine sentiment category
    if compound_score >= 0.05:
        sentiment = "positive"
    elif compound_score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Extract emotion words from VADER's internal lexicon
    words = text.lower().split()
    emotion_words = []
    word_scores = []

    # Access VADER's lexicon to find emotion words in the text
    for word in words:
        # Remove punctuation
        clean_word = word.strip('.,!?;:')
        # Check if word exists in VADER's lexicon
        if clean_word in vader_analyzer.lexicon:
            score = vader_analyzer.lexicon[clean_word]
            # Consider words with significant sentiment (|score| > 0.5) as emotion words
            if abs(score) > 0.5:
                emotion_words.append(clean_word)
                word_scores.append(score)

    return {
        "emotion_words": emotion_words,
        "word_scores": word_scores,
        "overall_score": compound_score,
        "sentiment": sentiment,
        "detailed_scores": vader_scores  # Include detailed VADER scores (neg, neu, pos, compound)
    }


def affective_ratio(data, domain="Movie Titles"):
    if os.path.exists(f"C4Styli/affective_{domain.lower().replace(' ', '_')}.json"):
        data = json.load(open(f"C4Styli/affective_{domain.lower().replace(' ', '_')}.json", "r", encoding="utf-8"))
        all_results = data["all"]
        year_results = data["year"]
    else:
        all_results = {"CN": [], "HK": []}
        year_results = {"CN": defaultdict(list), "HK": defaultdict(list)}
        
        for item in tqdm(data, total=len(data), desc="Processing Titles"):
            if domain == "Movie Titles":
                cn_text = item["title_cn"]
                hk_text = norm_text(item["title_hk"], "CN")
                cn_density = emotion_density(cn_text)
                hk_density = emotion_density(hk_text)
                all_results["CN"].append(cn_density)
                all_results["HK"].append(hk_density)
                year = item.get("year", -1)
                if year != -1:
                    y_bin = year_to_bin(year)
                    year_results["CN"][y_bin].append(cn_density)
                    year_results["HK"][y_bin].append(hk_density)
            elif domain == "Advertising Slogans":
                text = norm_text(item["slogan"], "CN")
                region = item["region"]
                density = emotion_density(text)
                all_results[region].append(density)
                year = item.get("year", -1)
                if year != -1:
                    y_bin = year_to_bin(year)
                    year_results[region][y_bin].append(density)
            else:
                raise ValueError(f"Invalid domain: {domain}")
        
        # with open(f"C4Styli/affective_{domain.lower().replace(' ', '_')}.json", "w", encoding="utf-8") as f:
        #     json.dump({"all": all_results, "year": year_results}, f, ensure_ascii=False, indent=4)
    
    print(f"HK affective ratio {np.mean(all_results['HK'])}")
    print(f"CN affective ratio {np.mean(all_results['CN'])}")
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    cn_means = {}
    hk_means = {}
    
    for year_bin, distances in year_results["CN"].items():
        cn_means[year_bin] = np.mean(distances)
    
    for year_bin, distances in year_results["HK"].items():
        hk_means[year_bin] = np.mean(distances)
    
    all_bins = sorted(set(cn_means.keys()) | set(hk_means.keys()))
    
    cn_values = [cn_means.get(bin, 0) for bin in all_bins]
    hk_values = [hk_means.get(bin, 0) for bin in all_bins]
        
    y_label = "Affective Ratio"
    figure_name = f"Affective Ratio by Time Period"
    save_path = f"C4Styli/affective_ratio_by_year_{domain}.pdf"
    if domain == "Movie Titles":
        figure_width = 6.5
        color_cn = '#5281C3'
        color_hk = '#BE514D'
    else:
        figure_width = 6
        color_cn = '#F8C452'
        color_hk = '#95C831'
    visualization(cn_values, hk_values, all_bins, y_label, figure_name, save_path, figure_width, color_cn, color_hk)
    
    print("\n统计信息:")
    print(f"CN - 时间周期数量: {len(cn_means)}")
    print(f"HK - 时间周期数量: {len(hk_means)}")
    print(f"CN - 平均距离范围: {min(cn_values):.3f} - {max(cn_values):.3f}")
    print(f"HK - 平均距离范围: {min(hk_values):.3f} - {max(hk_values):.3f}")


def reformulation_affective_ratio(data, domain="Movie Titles"):
    if os.path.exists(f"C4Styli/reformulation_affective_{domain.lower().replace(' ', '_')}.json"):
        data = json.load(open(f"C4Styli/reformulation_affective_{domain.lower().replace(' ', '_')}.json", "r", encoding="utf-8"))
        all_results = data["all"]
        year_results = data["year"]
    else:
        year_range = [t["year"] for t in titles if t["year"] != -1]
        max_year, min_year = max(year_range), min(year_range)
        MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"
        tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        model = MarianMTModel.from_pretrained(MODEL_NAME)
        model.eval()
        
        all_results = {"CN": [], "HK": []}
        year_results = {"CN": defaultdict(list), "HK": defaultdict(list)}
        
        for item in tqdm(titles, total=len(titles), desc="Processing Titles"):
            title_cn = item["title_cn"]
            cn_lit = translate_zh_to_en(title_cn, tokenizer, model)
            title_hk = norm_text(item["title_hk"], "CN")
            hk_lit = translate_zh_to_en(title_hk, tokenizer, model)
            
            title_en = item["title"]
            
            # Calculate emotion words and scores for English texts using VADER
            cn_emotion = extract_emotion_words_and_score(cn_lit)
            hk_emotion = extract_emotion_words_and_score(hk_lit)
            en_emotion = extract_emotion_words_and_score(title_en)

            # Use overall emotion score for affective calculation
            affective_cn = abs(cn_emotion["overall_score"])
            affective_hk = abs(hk_emotion["overall_score"])
            affective_en = abs(en_emotion["overall_score"])
            
            all_results["CN"].append(affective_cn - affective_en)
            all_results["HK"].append(affective_hk - affective_en)
            
            year = item["year"]
            if year != -1:
                y_bin = year_to_bin(year, 5, max_year, min_year)
                year_results["CN"][y_bin].append(affective_cn - affective_en)
                year_results["HK"][y_bin].append(affective_hk - affective_en)
                
        with open(f"C4Styli/reformulation_affective_{domain.lower().replace(' ', '_')}.json", "w", encoding="utf-8") as f:
            json.dump({"all": all_results, "year": year_results}, f, ensure_ascii=False, indent=4)
            
    print(f"HK reformulation affective ratio {np.mean(all_results['HK'])}")
    print(f"CN reformulation affective ratio {np.mean(all_results['CN'])}")
        
    cn_means = {}
    hk_means = {}
    
    for year_bin, distances in year_results["CN"].items():
        cn_means[year_bin] = np.mean(distances)
    
    for year_bin, distances in year_results["HK"].items():
        hk_means[year_bin] = np.mean(distances)
    
    all_bins = sorted(set(cn_means.keys()) | set(hk_means.keys()))
    
    cn_values = [cn_means.get(bin, 0) for bin in all_bins]
    hk_values = [hk_means.get(bin, 0) for bin in all_bins]
    
    y_label = "|Sent_trans| − |Sent_orig|"
    figure_name = f"Sentiment Expression Intensity by Time Period"
    save_path = f"C4Styli/reformulation_affective_ratio_by_year_{domain}.pdf"
    visualization(cn_values, hk_values, all_bins, y_label, figure_name, save_path, 6.5)
    
    print("\n统计信息:")
    print(f"CN - 时间周期数量: {len(cn_means)}")
    print(f"HK - 时间周期数量: {len(hk_means)}")
    print(f"CN - 平均距离范围: {min(cn_values):.3f} - {max(cn_values):.3f}")
    print(f"HK - 平均距离范围: {min(hk_values):.3f} - {max(hk_values):.3f}")

def compute_modifier_ratio(text: str) -> float:
    """
    Compute the ratio of (adjective + adverb) / noun in a text.
    Returns a float representing the proportion of modifier content relative to nouns.
    """
    if not text:
        return 0.0

    words = pseg.cut(text)
    adjective_count = 0
    adverb_count = 0
    noun_count = 0

    for word, pos in words:
        if pos in ['a', 'ag', 'an', 'b', 'ad', 'al', 'bl']:
            adjective_count += 1
        elif pos in ['d', 'dl']:
            adverb_count += 1
        elif pos in ['n', 'nr', 'ns', 'nt', 'nz', 'nw']:
            noun_count += 1
    modifier_count = adjective_count + adverb_count

    # Return ratio, handle division by zero
    return modifier_count / noun_count if noun_count > 0 else 0.0


def modifier_ratio(data: list[dict], domain="Movie Titles"):
    if os.path.exists(f"C4Styli/modifier_ration_{domain.lower().replace(' ', '_')}.json"):
        data = json.load(open(f"C4Styli/modifier_ration_{domain.lower().replace(' ', '_')}.json", "r", encoding="utf-8"))
        all_results = data["all"]
        year_results = data["year"]
    else:
        all_results = {"CN": [], "HK": []}
        year_results = {"CN": defaultdict(list), "HK": defaultdict(list)}
        
        for item in tqdm(data, total=len(data), desc="Processing Titles"):
            if domain == "Movie Titles":
                cn_text = item["title_cn"]
                hk_text = norm_text(item["title_hk"], "CN")
                cn_modifier = compute_modifier_ratio(cn_text)
                hk_modifier = compute_modifier_ratio(hk_text)
                all_results["CN"].append(cn_modifier)
                all_results["HK"].append(hk_modifier)
                year = item.get("year", -1)
                if year != -1:
                    y_bin = year_to_bin(year)
                    year_results["CN"][y_bin].append(cn_modifier)
                    year_results["HK"][y_bin].append(hk_modifier)
            elif domain == "Advertising Slogans":
                text = norm_text(item["slogan"], "CN")
                region = item["region"]
                modifier = compute_modifier_ratio(text)
                all_results[region].append(modifier)
                year = item.get("year", -1)
                if year != -1:
                    y_bin = year_to_bin(year)
                    year_results[region][y_bin].append(modifier)
            else:
                raise ValueError(f"Invalid domain: {domain}")
        
        with open(f"C4Styli/modifier_ration_{domain.lower().replace(' ', '_')}.json", "w", encoding="utf-8") as f:
            json.dump({"all": all_results, "year": year_results}, f, ensure_ascii=False, indent=4)
    
    print(f"HK modifier ratio {np.mean(all_results['HK'])}")
    print(f"CN modifier ratio {np.mean(all_results['CN'])}")
    
    cn_means = {}
    hk_means = {}
    
    for year_bin, distances in year_results["CN"].items():
        cn_means[year_bin] = np.mean(distances)
    
    for year_bin, distances in year_results["HK"].items():
        hk_means[year_bin] = np.mean(distances)
    
    all_bins = sorted(set(cn_means.keys()) | set(hk_means.keys()))
    
    cn_values = [cn_means.get(bin, 0) for bin in all_bins]
    hk_values = [hk_means.get(bin, 0) for bin in all_bins]
    
    y_label = "Modifier Ratio"
    figure_name = f"Modifier Ratio by Time Period"
    save_path = f"C4Styli/modifier_ratio_by_year_{domain}.pdf"
    if domain == "Movie Titles":
        figure_width = 6.5
        color_cn = '#5281C3'
        color_hk = '#BE514D'
    else:
        figure_width = 6
        color_cn = '#F8C452'
        color_hk = '#95C831'
    visualization(cn_values, hk_values, all_bins, y_label, figure_name, save_path, figure_width, color_cn, color_hk)
    
    print("\n统计信息:")
    print(f"CN - 时间周期数量: {len(cn_means)}")
    print(f"HK - 时间周期数量: {len(hk_means)}")
    print(f"CN - 平均距离范围: {min(cn_values):.3f} - {max(cn_values):.3f}")
    print(f"HK - 平均距离范围: {min(hk_values):.3f} - {max(hk_values):.3f}")


def visualization(cn_values, hk_values, all_bins, y_label, figure_name, save_path, figure_width=6.5, color_cn='#5281C3', color_hk='#BE514D'):
    # red-blue: #BE514D-#5281C3
    # yelllow-green: #F8C452-#95C831
    plt.figure(figsize=(figure_width, 4))
    x = range(len(all_bins))
    width = 0.35

    # Create bar plot
    plt.bar([i - width/2 for i in x], cn_values, width, label=f'CN', alpha=0.6, color=color_cn)
    plt.bar([i + width/2 for i in x], hk_values, width, label=f'HK', alpha=0.6, color=color_hk)

    # Add trend lines
    plt.plot([i - width/2 for i in x], cn_values, 'o-', color=color_cn,
             linewidth=2.5, markersize=4, alpha=0.9, markerfacecolor='white', markeredgewidth=2)
    plt.plot([i + width/2 for i in x], hk_values, 's--', color=color_hk,
             linewidth=2.5, markersize=4, alpha=0.9, markerfacecolor='white', markeredgewidth=2)

    # Set fonts
    times_font = font_manager.FontProperties(fname='C4Styli/Times-New-Roman-Bold.ttf')
    cambria_font = font_manager.FontProperties(fname='C4Styli/cambria-math.ttf')

    plt.xlabel('Time Period', fontproperties=times_font, fontsize=12)
    plt.ylabel(y_label, fontproperties=times_font, fontsize=12)
    plt.title(figure_name, fontproperties=times_font, fontsize=12)
    plt.xticks(x, all_bins, rotation=50, fontproperties=cambria_font, fontsize=10)

    for tick, xpos in zip(plt.gca().get_xticklabels(), x):
        tick.set_x(xpos - 0.2)

    plt.yticks(fontproperties=cambria_font, fontsize=10)
    plt.legend(prop=font_manager.FontProperties(fname='C4Styli/times.ttf', size=10), ncol=2)
    plt.grid(True, alpha=0.3)

    # Set y-axis limits
    min_y = min(min(cn_values), min(hk_values)) - 0.02
    max_y = max(max(cn_values), max(hk_values)) * 1.05
    plt.ylim(min_y, max_y)

    # Adjust x-axis margins
    plt.xlim(-0.5, len(all_bins) - 0.5)
    
    plt.margins(x=0.02)
    plt.tight_layout()

    # Save and show plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    print(f"图表已保存到: {save_path}")

def load_data():
    titles, slogans = [], []
    for m in ["finetune", "val", "probe"]:
        titles.extend(json.load(open(f"C4Styli/titles/{m}_movie_titles.json", "r", encoding="utf-8")))
        slogans.extend(json.load(open(f"C4Styli/slogans/{m}_advertise_slogans.json", "r", encoding="utf-8")))
    titles = [{
        "title": item["TITLE"],
        "title_cn": item["TITLE (CN)"],
        "title_hk": item["TITLE (HK)"],
        "year": int(item["YEAR"]),
    } for item in titles]
    
    slogans = [{
        "slogan": item["slogan"],
        "region": item["region"],
        "year": int(item["date"].split("-")[0]) if isinstance(item["date"], str) else int(item["date"]),
    } for item in slogans]
    titles = sorted(titles, key=lambda x: x["year"])
    slogans = sorted(slogans, key=lambda x: x["year"])
    
    return titles, slogans
    
if __name__ == '__main__':
    titles, slogans = load_data()
    # affective_ratio(titles, "Movie Titles")
    reformulation_affective_ratio(titles, "Movie Titles")
    # affective_ratio(slogans, "Advertising Slogans")
    english_ratio(slogans, "Advertising Slogans")
    modifier_ratio(titles, "Movie Titles")
    modifier_ratio(slogans, "Advertising Slogans")
    translation_distance(titles)