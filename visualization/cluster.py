#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text clustering analysis tool
function:
1. Load texts from file
2. Use LLM to infer and extract intermediate layer features
3. Use DBSCAN for clustering analysis
4. Visualize clustering results
"""

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Tuple, Union

from tqdm import trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from utils import *

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
font_prop = FontProperties(fname="./visualization/Times-New-Roman-Bold.ttf")


class TextClusterAnalyzer:
    """Text clustering analyzer"""

    def __init__(
        self,
        model_name: str = "/data/models/DeepSeek-R1-Distill-Qwen-14B",
        layer_indices: Union[int, List[int], str] = -2,
        merge_strategy: str = "concat",
        device: str = None,
    ):
        """
        Initialize analyzer

        Args:
            model_name: pretrained model name
            layer_indices: layer index or list of layer indices to extract features
                          (default: -2, can be single int like -2, list like [-1, -2, -3],
                          or "top_2_3" to automatically select the top 2/3 layers)
            merge_strategy: strategy to merge features from multiple layers
                          'concat': concatenate features (default)
                          'mean': average features
            device: running device ('cuda' or 'cpu')
            default is cuda if cuda is available, otherwise cpu
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        self.model, self.tokenizer = None, None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Optimize model loading: use float16/bfloat16, low_cpu_mem_usage, and device_map
        # Determine dtype based on device
        if self.device.type == "cuda":
            # Use bfloat16 if supported, otherwise float16
            try:
                if torch.cuda.is_bf16_supported():
                    self.torch_dtype = torch.bfloat16
                    print("[INFO] Using bfloat16 precision")
                else:
                    self.torch_dtype = torch.float16
                    print("[INFO] Using float16 precision")
            except:
                self.torch_dtype = torch.float16
                print("[INFO] Using float16 precision")
        else:
            self.torch_dtype = torch.float32
            print("[INFO] Using float32 precision (CPU)")

        self.model_name = model_name
        self.layer_indices = layer_indices
        self.merge_strategy = merge_strategy

        self.model, self.tokenizer = None, None

    def extract_features(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Extract features using LLM

        Args:
            texts: List of texts
            batch_size: batch size

        Returns:
            Feature matrix (n_samples, hidden_size or hidden_size * num_layers)
        """
        if self.model is None or self.tokenizer is None:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                output_hidden_states=True,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device.type == "cuda" else None,
            )

            # If device_map wasn't used, manually move to device
            if self.device.type == "cuda" and not hasattr(self.model, "hf_device_map"):
                self.model.to(self.device)

            self.model.eval()

            # Get total number of layers
            num_layers = self.model.config.num_hidden_layers
            print(f"[INFO] Model has {num_layers} layers (0 to {num_layers-1})")

            # Handle special "top_2_3" case
            if isinstance(layer_indices, str) and layer_indices == "top_2_3":
                # Calculate top 2/3 layers (from the last layer)
                num_top_layers = int(
                    num_layers
                    * int(layer_indices.split("_")[1])
                    / int(layer_indices.split("_")[2])
                )
                # Select layers from -1 to -num_top_layers (e.g., for 30 layers: -1 to -20)
                self.layer_indices = list(range(-15, -num_top_layers - 1, -1))
                print(
                    f"[INFO] Using 'top_2_3' strategy: selecting top {num_top_layers} layers"
                )
                print(f"[INFO] Selected layer indices: {self.layer_indices}")
            # Convert single int to list for uniform processing
            elif isinstance(layer_indices, int):
                self.layer_indices = [layer_indices]
            else:
                self.layer_indices = layer_indices

            self.merge_strategy = merge_strategy

            if len(self.layer_indices) == 1:
                print(f"[INFO] Extracting features from layer {self.layer_indices[0]}")
            else:
                print(
                    f"[INFO] Extracting features from {len(self.layer_indices)} layers with '{merge_strategy}' strategy"
                )
        features = []

        print(f"Extracting features (batch size: {batch_size})...")

        with torch.no_grad():
            for i in trange(0, len(texts), batch_size, desc="Extracting features"):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    [text["input_text"] for text in batch_texts],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(**inputs)

                # extract the hidden state of the output_text part
                batch_feature_list = []
                for idx, text_type in enumerate(
                    [text["region"] for text in batch_texts]
                ):
                    orig_text = batch_texts[idx]
                    output_text = orig_text["output_text"]
                    prefix = orig_text["input_text"]
                    # 计算output_text在token中的起止位 置
                    encoded_prefix = self.tokenizer(prefix, add_special_tokens=False)
                    encoded_all = self.tokenizer(
                        prefix + output_text, add_special_tokens=False
                    )
                    prefix_len = len(encoded_prefix["input_ids"])
                    output_len = len(
                        self.tokenizer(output_text, add_special_tokens=False)[
                            "input_ids"
                        ]
                    )
                    # obtain the hidden states（each layer is: [batch, seq, dim]）
                    layer_outputs = []
                    for layer_idx in self.layer_indices:
                        # [batch, seq, dim]
                        hidden_states = outputs.hidden_states[layer_idx][idx]
                        seq_len = hidden_states.shape[0]

                        # Handle edge cases: empty output_text or index out of bounds
                        if output_len == 0:
                            # If output_text is empty, use the last token of prefix as fallback
                            if prefix_len > 0:
                                output_hidden = hidden_states[
                                    prefix_len - 1 : prefix_len, :
                                ]
                            else:
                                # If prefix is also empty, use the first token
                                output_hidden = hidden_states[0:1, :]
                        elif prefix_len + output_len > seq_len:
                            # If index out of bounds, use available tokens
                            end_idx = min(prefix_len + output_len, seq_len)
                            if prefix_len < seq_len:
                                output_hidden = hidden_states[prefix_len:end_idx, :]
                            else:
                                # If prefix_len >= seq_len, use the last token
                                output_hidden = hidden_states[-1:, :]
                        else:
                            # Normal case: extract the output_text part
                            output_hidden = hidden_states[
                                prefix_len : prefix_len + output_len, :
                            ]

                        # mean pooling
                        # Convert to float32 before numpy conversion (numpy doesn't support bfloat16)
                        if output_hidden.shape[0] == 0:
                            # Fallback: use zero vector if still empty
                            output_feat = (
                                torch.zeros(output_hidden.shape[1])
                                .float()
                                .cpu()
                                .numpy()
                            )
                        else:
                            output_feat = (
                                output_hidden.mean(dim=0).float().cpu().numpy()
                            )  # shape: [dim]
                        layer_outputs.append(output_feat)
                    # merge the features
                    if len(layer_outputs) == 1:
                        feat = layer_outputs[0]
                    elif self.merge_strategy == "concat":
                        feat = np.concatenate(layer_outputs, axis=-1)
                    elif self.merge_strategy == "mean":
                        feat = np.mean(np.stack(layer_outputs, axis=0), axis=0)
                    else:
                        raise ValueError(
                            f"Unknown merge strategy: {self.merge_strategy}"
                        )
                    batch_feature_list.append(feat)
                # add the two parts together
                features.append(np.stack(batch_feature_list, axis=0))

        features = np.vstack(features)
        print(f"[INFO] Feature extraction completed, shape: {features.shape}")

        # Check and handle NaN values
        nan_mask = np.isnan(features).any(axis=1)
        nan_count = nan_mask.sum()
        if nan_count > 0:
            print(
                f"[WARNING] Found {nan_count} samples with NaN values, replacing with zeros"
            )
            # Replace NaN with zeros
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for infinite values
        inf_mask = np.isinf(features).any(axis=1)
        inf_count = inf_mask.sum()
        if inf_count > 0:
            print(
                f"[WARNING] Found {inf_count} samples with infinite values, replacing with zeros"
            )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def cluster_dbscan(
        self, features: np.ndarray, eps: float = 0.5, min_samples: int = 2
    ) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """
        Use DBSCAN for clustering

        Args:
            features: feature matrix
            eps: DBSCAN neighborhood radius
            min_samples: minimum number of samples in a core point

        Returns:
            (clustering labels, statistics)
        """
        print(
            f"[INFO] Using DBSCAN for clustering (eps={eps}, min_samples={min_samples})..."
        )

        # Final check and clean NaN/inf values before DBSCAN
        if np.isnan(features).any() or np.isinf(features).any():
            print("[WARNING] Cleaning NaN/inf values before DBSCAN")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Use DBSCAN for clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = dbscan.fit_predict(features)

        # Statistics
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        stats = {"n_clusters": n_clusters, "n_noise": n_noise, "n_samples": len(labels)}

        print(f"[INFO] Clustering completed:")
        print(f"  - number of clusters: {n_clusters}")
        print(f"  - number of noise points: {n_noise}")
        print(f"  - total number of samples: {stats['n_samples']}")

        # Statistics of each cluster
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique_labels) > 0:
            print(
                f"  - size of each cluster: {dict(zip(unique_labels.tolist(), counts.tolist()))}"
            )

        return labels, stats, features

    def visualize(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        texts: List[dict],
        output_path: str = None,
    ):
        """
        Visualize clustering results

        Args:
            features: feature matrix
            labels: clustering labels
            texts: original texts (should be List[dict], each dict包含原文本等信息)
            output_path: output image path
        """
        print("[INFO] Generating visualization...")

        # Final cleanup of NaN and inf values before t-SNE
        # Make a copy to avoid modifying the original array
        features_clean = features.copy()

        nan_mask = np.isnan(features_clean).any(axis=1)
        inf_mask = np.isinf(features_clean).any(axis=1)
        nan_count = nan_mask.sum()
        inf_count = inf_mask.sum()

        if nan_count > 0 or inf_count > 0:
            print(
                f"[WARNING] Found {nan_count} NaN and {inf_count} inf values before t-SNE, cleaning..."
            )
            # Clean NaN and inf values
            features_clean = np.nan_to_num(
                features_clean, nan=0.0, posinf=0.0, neginf=0.0
            )

            # Verify cleanup - check element by element if needed
            remaining_nan = np.isnan(features_clean).sum()
            remaining_inf = np.isinf(features_clean).sum()
            if remaining_nan > 0 or remaining_inf > 0:
                print(
                    f"[ERROR] Still found {remaining_nan} NaN and {remaining_inf} inf values, forcing cleanup..."
                )
                # Force replace any remaining problematic values
                features_clean[np.isnan(features_clean)] = 0.0
                features_clean[np.isinf(features_clean)] = 0.0

        # Use cleaned features for t-SNE
        features = features_clean

        # Additional safety check and cleanup for t-SNE
        if not np.all(np.isfinite(features)):
            print(
                "[WARNING] Features still contain non-finite values after cleanup, performing final sanitization..."
            )
            # Final sanitization: replace any remaining non-finite values with zeros
            features = np.where(np.isfinite(features), features, 0.0)

            # Verify the data is now clean
            if not np.all(np.isfinite(features)):
                raise ValueError(
                    "Failed to clean non-finite values from features array"
                )

        print(
            f"[INFO] Features shape after cleanup: {features.shape}, finite check: {np.all(np.isfinite(features))}"
        )

        # Check if features have variance (t-SNE needs this)
        feature_variance = np.var(features, axis=0)
        zero_variance_cols = np.sum(feature_variance == 0)
        if zero_variance_cols > 0:
            print(
                f"[WARNING] {zero_variance_cols} features have zero variance, this may affect t-SNE performance"
            )

        type_labels = [
            text["region"] + "_" + text["domain"] for text in texts
        ]  # "CN" or "HK"

        # Final comprehensive cleanup before t-SNE
        print("[INFO] Performing final cleanup before t-SNE...")
        features = np.nan_to_num(features, nan=0.0, posinf=1e-6, neginf=-1e-6)

        # Ensure minimum variance for t-SNE stability
        feature_std = np.std(features, axis=0)
        zero_std_mask = feature_std == 0
        if np.any(zero_std_mask):
            print(
                f"[WARNING] {np.sum(zero_std_mask)} features have zero standard deviation, adding small noise..."
            )
            # Add tiny noise to zero-variance features
            noise = np.random.normal(0, 1e-6, features.shape)
            features[:, zero_std_mask] += noise[:, zero_std_mask]

        # Final verification
        if not np.all(np.isfinite(features)):
            print(
                "[ERROR] Features still contain non-finite values after final cleanup!"
            )
            features = np.where(np.isfinite(features), features, 0.0)

        print(
            f"[INFO] Final features check - finite: {np.all(np.isfinite(features))}, shape: {features.shape}"
        )

        # First, perform PCA manually to reduce dimensionality before t-SNE
        print("[INFO] Performing manual PCA preprocessing before t-SNE...")

        # Reduce to reasonable dimensionality first (e.g., 50 components)
        n_pca_components = min(50, features.shape[1], len(features) - 1)
        pca = PCA(n_components=n_pca_components, random_state=42)

        try:
            features_pca = pca.fit_transform(features)
        except Exception as e:
            features_pca = pca.fit_transform(features)

        # Ensure PCA output is clean and has proper scale
        features_pca = np.nan_to_num(features_pca, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale the PCA output to avoid numerical issues
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        features_pca = scaler.fit_transform(features_pca)

        print(
            f"[INFO] PCA completed, reduced to {n_pca_components} dimensions, shape: {features_pca.shape}"
        )

        # t-SNE dimensionality reduction (without internal PCA)
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, max(5, len(features_pca) - 1)),
            init="random",  # Use random initialization instead of PCA
            learning_rate="auto",  # Let t-SNE choose optimal learning rate
        )

        # Final check before t-SNE
        print(
            f"[INFO] t-SNE input check - finite: {np.all(np.isfinite(features_pca))}, shape: {features_pca.shape}"
        )

        print("[INFO] Starting t-SNE...")
        features_2d = tsne.fit_transform(features_pca)

        # Check t-SNE output for any issues
        if not np.all(np.isfinite(features_2d)):
            print("[WARNING] t-SNE output contains non-finite values, cleaning...")
            features_2d = np.nan_to_num(features_2d, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"[INFO] t-SNE completed successfully, output shape: {features_2d.shape}")

        plt.figure(figsize=(8, 6))

        # type/color for CN/HK
        unique_types = sorted(set(type_labels))
        type_color_map = {t: plt.cm.tab10(i) for i, t in enumerate(unique_types)}
        # cluster/marker
        unique_clusters = sorted(set(labels))
        # assign marker for each cluster
        # Enough for very many clusters: many markers from matplotlib, then fallback to 'o'
        marker_list = [
            "o",  # circle
            "s",  # square
            "^",  # triangle up
            "v",  # triangle down
            "D",  # diamond
            "*",  # star
            "x",  # x
            "+",  # plus
            ".",  # point
        ]
        cluster_marker_map = {
            l: marker_list[i % len(marker_list)] for i, l in enumerate(unique_clusters)
        }

        type_label_zh = {"CN": "CN", "HK": "HK"}

        # Plot points, grouping legend per (type,cluster)
        print(f"[INFO] Starting to plot {len(features_2d)} points...")
        legend_done = set()
        for idx in range(len(features_2d)):
            t = type_labels[idx]  # "CN" or "HK"
            label = labels[idx]
            # Color: by type (region)，Marker: by cluster
            color = type_color_map.get(t, (0.3, 0.3, 0.3)) if label != -1 else "black"
            marker = cluster_marker_map.get(label, "o")
            # Legend: only one per type/cluster
            if label == -1:
                legend_label = f"noise-{type_label_zh.get(t, t)}"
            else:
                legend_label = f"cluster {label}-{type_label_zh.get(t, t)}"
            draw_kwargs = {}
            if (label, t) not in legend_done:
                draw_kwargs["label"] = legend_label
                legend_done.add((label, t))
            plt.scatter(
                features_2d[idx, 0],
                features_2d[idx, 1],
                c=[color],
                marker=marker,
                s=100,
                alpha=0.7,
                edgecolors="k" if label != -1 else "gray",
                linewidth=1.0,
                **draw_kwargs,
            )
        plt.title(
            "DBSCAN text clustering results (t-SNE, color: CN/HK, shape: cluster)",
            fontsize=14,
        )
        plt.xlabel("t-SNE dimension 1", fontsize=12, fontproperties=font_prop)
        plt.ylabel("t-SNE dimension 2", fontsize=12, fontproperties=font_prop)
        plt.tight_layout()
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=11,
        )

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[INFO] Visualization results saved to: {output_path}")
        else:
            plt.savefig("cluster_result.png", dpi=300, bbox_inches="tight")
            print("[INFO] Visualization results saved to: cluster_result.png")
        plt.close()

    def save_results(
        self,
        texts: List[Dict],
        labels: np.ndarray,
        output_path: str = "cluster_results.json",
    ):
        """
        保存聚类结果，将 CN 和 HK 的 cluster 分别记录

        Args:
            texts: 文本数据列表（每个元素包含 input_text/output_text/region）
            labels: 聚类标签数组（与 texts 一一对应）
            output_path: 输出文件路径
        """
        if len(labels) != len(texts):
            raise ValueError(
                f"Labels length {len(labels)} does not match texts length {len(texts)}"
            )

        # Pair HK and CN records (data order: HK, CN, HK, CN, ...)
        # Group texts by base input_text to pair CN and HK records
        grouped_data = {}
        for idx, text in enumerate(texts):
            # Extract the base prompt (before region-specific part)
            input_text = text["input_text"]
            # For movie data, the region is in the prompt, extract the base part
            # The format is: "...目标地区：中国{region}" where region is "大陆" or "香港"
            if "目标地区" in input_text:
                base_key = input_text.split("目标地区")[0]
            else:
                # For other data types, use input_text as key but remove region-specific parts
                base_key = input_text

            region = text["region"]
            if base_key not in grouped_data:
                grouped_data[base_key] = {}

            # Store both CN and HK records for each base_key
            if region not in grouped_data[base_key]:
                grouped_data[base_key][region] = {
                    "input_text": input_text,
                    "output_text": text["output_text"],
                    "cluster": int(labels[idx]),
                }

        # Convert grouped data to results format
        results = []
        for base_key, regions in grouped_data.items():
            cn_data = regions.get("CN")
            hk_data = regions.get("HK")

            # Use the input_text from CN or HK (they should be similar except for region)
            if cn_data:
                input_text = cn_data["input_text"]
                # Replace region-specific part for cleaner output
                if "目标地区" in input_text:
                    input_text = input_text.replace("目标地区：中国大陆", "目标地区：中国{region}")
            elif hk_data:
                input_text = hk_data["input_text"]
                if "目标地区" in input_text:
                    input_text = input_text.replace("目标地区：中国香港", "目标地区：中国{region}")
            else:
                input_text = base_key

            results.append(
                {
                    "input_text": input_text,
                    "cn_output_text": cn_data.get("output_text", "") if cn_data else "",
                    "cn_cluster": cn_data.get("cluster", -1) if cn_data else -1,
                    "hk_output_text": hk_data.get("output_text", "") if hk_data else "",
                    "hk_cluster": hk_data.get("cluster", -1) if hk_data else -1,
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 聚类结果已保存到: {output_path}（包含 CN 和 HK 分别的 cluster）")

    def analyze(
        self,
        texts: List[Dict],
        eps: float = 0.5,
        min_samples: int = 2,
        batch_size: int = 8,
        output_prefix: str = "output",
    ):
        """
        完整的分析流程

        Args:
            file_path: 输入文件路径
            eps: DBSCAN的邻域半径
            min_samples: 核心点的最小样本数
            batch_size: 批次大小
            output_prefix: 输出文件前缀
        """
        print("=" * 60)
        print("[INFO] Starting text clustering analysis...")
        print("=" * 60)

        if len(texts) == 0:
            print("[ERROR] No texts found")
            return

        # 1. Extract features
        if os.path.exists(f"{output_prefix}_features.npy"):
            features = np.load(f"{output_prefix}_features.npy")
            print(f"[INFO] Features loaded from: {output_prefix}_features.npy")
        else:
            features = self.extract_features(texts, batch_size=batch_size)
            np.save(f"{output_prefix}_features.npy", features)
            print(f"[INFO] Features saved to: {output_prefix}_features.npy")
        print(f"[INFO] Features saved to: {output_prefix}_features.npy")

        # Clean features before processing
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        if nan_count > 0 or inf_count > 0:
            print(
                f"[WARNING] Cleaning {nan_count} NaN and {inf_count} inf values from features before clustering"
            )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Use DBSCAN for clustering
        labels, stats, features_clean = self.cluster_dbscan(
            features, eps=eps, min_samples=min_samples
        )

        # 3. Visualize
        self.visualize(
            features_clean, labels, texts, f"{output_prefix}_visualization.png"
        )

        # 4. Save results
        self.save_results(texts, labels, f"{output_prefix}_results.json")

        print("=" * 60)
        print("[INFO] Analysis completed!")
        print("=" * 60)

        return {"texts": texts, "features": features, "labels": labels, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description="Text clustering analysis tool")
    parser.add_argument(
        "--model",
        type=str,
        default="/data/models/DeepSeek-R1-Distill-Qwen-14B",
        help="pretrained model name (default: /data/models/DeepSeek-R1-Distill-Qwen-7B )",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="top_2_3",
        help="layer index or indices to extract features (default: -2). "
        "Use comma-separated values for multiple layers, e.g., '-1,-2,-3'. "
        "You can also use 'top_2_3' to select the top 2/3 of layers starting from the last layer (e.g., for a 30-layer model, 'top_2_3' will select layers -1 to -20). "
        "If 'top_2_3' is specified, the actual layer indices will be computed internally based on the model architecture.",
    )
    parser.add_argument(
        "--merge-strategy",
        type=str,
        default="mean",
        choices=["concat", "mean"],
        help="strategy to merge features from multiple layers: 'concat' or 'mean' (default: concat)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.15,
        help="DBSCAN neighborhood radius (default: 0.5)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=12,
        help="minimum number of samples in a core point (default: 2)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size (default: 8)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="output",
        help="output file prefix (default: output)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device (cuda/cpu, default is cuda if cuda is available)",  # default is cuda if cuda is available
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="both",
        choices=["title", "slogan", "both"],
        help="data type (default: title)",
    )

    args = parser.parse_args()

    # Parse layer indices
    layer_indices_str = args.layers.strip()
    if layer_indices_str == "top_2_3":
        # Special case: use top 2/3 layers
        layer_indices = "top_2_3"
    elif "," in layer_indices_str:
        # Multiple layers
        layer_indices = [int(idx.strip()) for idx in layer_indices_str.split(",")]
    else:
        # Single layer (as integer)
        try:
            layer_indices = int(layer_indices_str)
        except ValueError:
            raise ValueError(
                f"Invalid layer specification: {layer_indices_str}. "
                "Use an integer (e.g., -2), comma-separated integers (e.g., -1,-2,-3), or 'top_2_3'"
            )

    if args.data_type == "title":
        texts = load_movie_data("C4Styli/titles/val_movie_titles.json")
    elif args.data_type == "slogan":
        texts = load_advertising_data("C4Styli/slogans/val_advertise_slogans.json")
    elif args.data_type == "both":
        texts = load_movie_data(
            "C4Styli/titles/val_movie_titles.json"
        ) + load_advertising_data("C4Styli/slogans/val_advertise_slogans.json")
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    # create analyzer
    analyzer = TextClusterAnalyzer(
        model_name=args.model,
        layer_indices=layer_indices,
        merge_strategy=args.merge_strategy,
        device=args.device,
    )

    # 执行分析
    analyzer.analyze(
        texts=texts,
        eps=args.eps,
        min_samples=args.min_samples,
        batch_size=args.batch_size,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
