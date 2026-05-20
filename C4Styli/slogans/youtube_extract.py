#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube视频批量数据抓取工具
支持获取视频标题、描述、时长等信息，以及提取字幕内容
"""

import json
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import dashscope
import yt_dlp
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

try:
    from pytube import YouTube

    PYTUBE_AVAILABLE = True
except ImportError:
    PYTUBE_AVAILABLE = False

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


class YouTubeExtractor:
    """YouTube视频数据提取器"""

    def __init__(
        self,
        output_dir: str = "output",
        retry_times: int = 3,
        delay: float = 1.0,
        subtitle_delay: float = 0.5,
        skip_subtitles: bool = False,
        use_proxy: bool = False,
        proxy_url: str = None,
    ):
        """
        初始化提取器

        Args:
            output_dir: 输出目录
            retry_times: 重试次数
            delay: 请求间隔延迟
            subtitle_delay: 字幕下载延迟倍数
            skip_subtitles: 是否跳过字幕下载
            use_proxy: 是否使用代理
            proxy_url: 代理URL
        """
        self.output_dir = output_dir
        self.retry_times = retry_times
        self.delay = delay
        self.subtitle_delay = subtitle_delay
        self.skip_subtitles = skip_subtitles
        self.use_proxy = use_proxy
        self.proxy_url = proxy_url

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # yt-dlp 配置
        self.ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "writesubtitles": not self.skip_subtitles,
            "writeautomaticsub": not self.skip_subtitles,
            "subtitleslangs": ["zh-hk", "zh", "zh-Hans", "zh-Hant", "yue", "en"],
            "subtitlesformat": "vtt",
            "skip_download": True,  # 不下载视频，只获取信息和字幕
            "outtmpl": os.path.join(self.output_dir, "%(id)s.%(ext)s"),  # 字幕文件输出模板
        }

    def extract_video_info(self, url: str) -> Optional[Dict]:
        """
        提取单个YouTube视频的基本信息

        Args:
            url: YouTube视频URL

        Returns:
            包含视频信息的字典，如果失败返回None
        """
        for attempt in range(self.retry_times):
            try:
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    # 提取视频信息
                    info = ydl.extract_info(url, download=False)
                    if info.get("duration", 0) > 30:
                        return None

                    if not info:
                        print(f"无法获取视频信息: {url}")
                        return None

                    # 提取基本信息
                    video_info = {
                        "video_id": info.get("id", ""),
                        "title": info.get("title", ""),
                        "description": info.get("description", ""),
                        "duration": info.get("duration", 0),  # 时长（秒）
                        "view_count": info.get("view_count", 0),
                        "like_count": info.get("like_count", 0),
                        "upload_date": info.get("upload_date", ""),
                        "uploader": info.get("uploader", ""),
                        "uploader_id": info.get("uploader_id", ""),
                        "channel_url": info.get("channel_url", ""),
                        "tags": info.get("tags", []),
                        "categories": info.get("categories", []),
                        "url": url,
                        "subtitles": {},
                    }

                    # 提取字幕信息 - 使用yt-dlp下载到本地然后读取
                    video_info["subtitles"] = {}

                    if not self.skip_subtitles:
                        print(f"正在尝试提取字幕...")

                        # 策略1: 使用yt-dlp直接下载字幕
                        try:
                            subtitle_opts = self.ydl_opts.copy()
                            subtitle_opts["skip_download"] = False
                            subtitle_opts["write_info_json"] = False
                            subtitle_opts["outtmpl"] = os.path.join(
                                self.output_dir, f"{info['id']}.%(ext)s"
                            )

                            with yt_dlp.YoutubeDL(subtitle_opts) as ydl_sub:
                                ydl_sub.download([url])

                            video_info["subtitles"] = self._read_local_subtitles(
                                info["id"]
                            )

                        except Exception as e:
                            print(f"yt-dlp字幕下载失败: {str(e)}")

                        # 策略2: 如果没有字幕，回退到手动HTTP下载
                        if not video_info["subtitles"]:
                            print("尝试手动HTTP下载字幕...")
                            if "subtitles" in info:
                                manual_subs = self.extract_subtitles_with_retry(
                                    info["subtitles"]
                                )
                                video_info["subtitles"].update(manual_subs)

                            if (
                                "automatic_captions" in info
                                and not video_info["subtitles"]
                            ):
                                auto_subs = self.extract_subtitles_with_retry(
                                    info["automatic_captions"]
                                )
                                video_info["subtitles"].update(auto_subs)

                        # 策略3: 如果仍然没有字幕，尝试浏览器模拟
                        if not video_info["subtitles"]:
                            print("尝试浏览器模拟提取字幕...")
                            browser_subs = (
                                self.extract_subtitles_with_browser_simulation(url)
                            )
                            video_info["subtitles"].update(browser_subs)

                        # 策略4: 最终尝试使用youtube-transcript-api
                        if not video_info["subtitles"]:
                            print("尝试使用youtube-transcript-api...")
                            transcript_subs = (
                                self.extract_subtitles_with_transcript_api(info["id"])
                            )
                            video_info["subtitles"].update(transcript_subs)

                        # 策略5: 如果仍然失败，尝试使用pytube
                        if not video_info["subtitles"] and PYTUBE_AVAILABLE:
                            print("尝试使用pytube提取字幕...")
                            try:
                                pytube_subs = self.extract_subtitles_with_pytube(url)
                                video_info["subtitles"].update(pytube_subs)
                            except Exception as e:
                                print(f"pytube方法出错: {str(e)}")

                        if video_info["subtitles"]:
                            print(f"✓ 成功提取 {len(video_info['subtitles'])} 种语言的字幕")
                        else:
                            print("✗ 所有字幕提取方法均失败")

                    print(f"成功提取视频: {video_info['title']}")
                    return video_info

            except Exception as e:
                print(f"提取失败 (尝试 {attempt + 1}/{self.retry_times}): {url} - {str(e)}")
                if attempt < self.retry_times - 1:
                    time.sleep(self.delay * (attempt + 1))  # 递增延迟

        return None

    def _extract_subtitle_content(self, subtitles_dict: Dict) -> Dict[str, str]:
        """
        从yt-dlp的字幕字典中提取字幕文本内容

        Args:
            subtitles_dict: yt-dlp返回的字幕字典

        Returns:
            语言代码到字幕文本的映射
        """
        subtitle_content = {}

        for lang, subtitle_list in subtitles_dict.items():
            if not subtitle_list:
                continue

            # 选择最佳格式（优先vtt，然后srt）
            best_subtitle = None
            for sub in subtitle_list:
                if sub.get("ext") == "vtt":
                    best_subtitle = sub
                    break
                elif sub.get("ext") == "srt" and not best_subtitle:
                    best_subtitle = sub

            if best_subtitle and "url" in best_subtitle:
                # 重试下载字幕
                for attempt in range(self.retry_times):
                    try:
                        import time

                        import requests

                        # 添加字幕下载延迟
                        time.sleep(self.delay * self.subtitle_delay)

                        # 使用更真实的浏览器头
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                            "Accept-Encoding": "gzip, deflate, br",
                            "Connection": "keep-alive",
                            "Upgrade-Insecure-Requests": "1",
                            "Sec-Fetch-Dest": "document",
                            "Sec-Fetch-Mode": "navigate",
                            "Sec-Fetch-Site": "none",
                            "Cache-Control": "max-age=0",
                            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                            "sec-ch-ua-mobile": "?0",
                            "sec-ch-ua-platform": '"Windows"',
                        }

                        # 设置代理
                        proxies = None
                        if self.use_proxy and self.proxy_url:
                            proxies = {"http": self.proxy_url, "https": self.proxy_url}

                        response = requests.get(
                            best_subtitle["url"],
                            headers=headers,
                            proxies=proxies,
                            timeout=30,  # 增加超时时间
                        )
                        response.raise_for_status()

                        # 解析VTT或SRT格式
                        content = self._parse_subtitle_text(
                            response.text, best_subtitle["ext"]
                        )
                        if content.strip():
                            subtitle_content[lang] = content
                            break  # 成功下载，跳出重试循环

                    except requests.exceptions.HTTPError as e:
                        if response.status_code == 429:
                            # 请求过于频繁，增加等待时间
                            wait_time = self.delay * (attempt + 2)
                            print(
                                f"字幕下载限速 {lang}，等待 {wait_time} 秒后重试 ({attempt + 1}/{self.retry_times})"
                            )
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"字幕下载HTTP错误 {lang}: {str(e)}")
                            break
                    except Exception as e:
                        print(
                            f"下载字幕失败 {lang} (尝试 {attempt + 1}/{self.retry_times}): {str(e)}"
                        )
                        if attempt < self.retry_times - 1:
                            time.sleep(self.delay)
                            continue
                        else:
                            break

        return subtitle_content

    def extract_subtitles_with_retry(
        self, subtitles_dict: Dict, preferred_langs: List[str] = None
    ) -> Dict[str, str]:
        """
        智能提取字幕，优先下载指定的语言，并带有重试机制

        Args:
            subtitles_dict: yt-dlp返回的字幕字典
            preferred_langs: 优先语言列表

        Returns:
            语言代码到字幕文本的映射
        """
        if not preferred_langs:
            preferred_langs = ["zh-hk", "zh", "zh-Hans", "zh-Hant", "yue", "en"]

        subtitle_content = {}

        # 优先处理指定的语言
        for lang in preferred_langs:
            if lang in subtitles_dict and subtitles_dict[lang]:
                subtitle_list = subtitles_dict[lang]

                # 选择最佳格式（优先vtt，然后srt）
                best_subtitle = None
                for sub in subtitle_list:
                    if sub.get("ext") == "vtt":
                        best_subtitle = sub
                        break
                    elif sub.get("ext") == "srt" and not best_subtitle:
                        best_subtitle = sub

                if best_subtitle and "url" in best_subtitle:
                    content = self._download_subtitle_with_retry(best_subtitle, lang)
                    if content and content.strip():
                        subtitle_content[lang] = content
                        print(f"✓ 成功下载字幕: {lang}")
                        break  # 只下载第一个可用的字幕

        return subtitle_content

    def _download_subtitle_with_retry(
        self, subtitle_info: Dict, lang: str
    ) -> Optional[str]:
        """
        带重试机制的字幕下载

        Args:
            subtitle_info: 字幕信息字典
            lang: 语言代码

        Returns:
            字幕文本内容，如果失败返回None
        """
        import time

        import requests

        for attempt in range(self.retry_times):
            try:
                # 添加字幕下载延迟
                time.sleep(self.delay * self.subtitle_delay)

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }

                response = requests.get(
                    subtitle_info["url"], headers=headers, timeout=15
                )
                response.raise_for_status()

                # 解析VTT或SRT格式
                content = self._parse_subtitle_text(response.text, subtitle_info["ext"])
                return content.strip() if content else None

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    # 请求过于频繁，增加等待时间
                    wait_time = self.delay * (attempt + 2)
                    print(
                        f"字幕下载限速 {lang}，等待 {wait_time:.1f} 秒后重试 ({attempt + 1}/{self.retry_times})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"字幕下载HTTP错误 {lang}: {response.status_code}")
                    return None
            except Exception as e:
                if attempt < self.retry_times - 1:
                    wait_time = self.delay * (attempt + 1)
                    print(
                        f"字幕下载失败 {lang}，等待 {wait_time:.1f} 秒后重试 ({attempt + 1}/{self.retry_times}): {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    print(f"字幕下载最终失败 {lang}: {str(e)}")
                    return None

        return None

    def _parse_subtitle_text(self, text: str, format_type: str) -> str:
        """
        解析字幕文本，提取纯文本内容

        Args:
            text: 原始字幕文本
            format_type: 字幕格式 ('vtt' 或 'srt')

        Returns:
            提取的纯文本内容
        """
        lines = text.split("\n")
        content_lines = []

        # 跳过VTT头部
        if format_type == "vtt":
            # 跳过WEBVTT头部
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip() == "":
                    start_idx = i + 1
                    break

        # 跳过SRT头部和时间戳
        skip_next = False
        for line in lines[start_idx:]:
            line = line.strip()

            # 跳过空行、数字编号、时间戳
            if not line or line.isdigit() or "-->" in line or line.startswith("WEBVTT"):
                skip_next = False
                continue

            # 跳过样式标签
            if line.startswith("<") and line.endswith(">"):
                continue

            # 跳过位置信息（如在VTT中）
            if any(x in line for x in ["align:", "position:", "size:", "vertical:"]):
                continue

            # 添加有效文本行
            if line:
                content_lines.append(line)

        # 合并文本并清理
        content = " ".join(content_lines)

        # 清理HTML标签和特殊字符
        content = re.sub(r"<[^>]+>", "", content)  # 移除HTML标签
        content = re.sub(r"&[^;]+;", "", content)  # 移除HTML实体
        content = re.sub(r"\s+", " ", content)  # 合并多个空格

        return content.strip()

    def extract_subtitles_with_browser_simulation(
        self, video_url: str, preferred_langs: List[str] = None
    ) -> Dict[str, str]:
        """
        使用浏览器模拟的方法提取字幕（备选方案）

        Args:
            video_url: YouTube视频URL
            preferred_langs: 优先语言列表

        Returns:
            语言代码到字幕文本的映射
        """
        try:
            # 尝试使用yt-dlp的浏览器模拟模式
            from yt_dlp import YoutubeDL

            browser_opts = {
                "quiet": True,
                "no_warnings": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": preferred_langs or ["zh-hk", "zh", "en"],
                "subtitlesformat": "vtt",
                "skip_download": True,
                "cookiesfrombrowser": None,  # 尝试从浏览器获取cookies
                "extractor_args": {
                    "youtube": {
                        "player_client": ["ios", "web"],
                        "player_skip": ["js"],
                    }
                },
            }

            with YoutubeDL(browser_opts) as ydl:
                # 先尝试获取字幕信息
                info = ydl.extract_info(video_url, download=False)

                if "subtitles" in info or "automatic_captions" in info:
                    # 下载字幕到临时位置
                    temp_dir = os.path.join(self.output_dir, "temp_subtitles")
                    os.makedirs(temp_dir, exist_ok=True)

                    download_opts = browser_opts.copy()
                    download_opts["outtmpl"] = os.path.join(temp_dir, "%(id)s.%(ext)s")

                    with YoutubeDL(download_opts) as ydl_download:
                        ydl_download.download([video_url])

                    # 读取下载的字幕
                    video_id = info.get("id", "")
                    subtitles = self._read_local_subtitles_from_dir(video_id, temp_dir)

                    # 清理临时文件
                    import shutil

                    shutil.rmtree(temp_dir, ignore_errors=True)

                    return subtitles

        except Exception as e:
            print(f"浏览器模拟字幕提取失败: {str(e)}")

        return {}

    def _read_local_subtitles_from_dir(
        self, video_id: str, temp_dir: str
    ) -> Dict[str, str]:
        """
        从临时目录读取字幕文件

        Args:
            video_id: 视频ID
            temp_dir: 临时目录

        Returns:
            语言代码到字幕文本的映射
        """
        subtitles = {}

        for filename in os.listdir(temp_dir):
            if filename.startswith(video_id) and filename.endswith(".vtt"):
                # 解析语言代码
                lang = "en"  # 默认
                if ".zh-hk." in filename:
                    lang = "zh-hk"
                elif ".zh." in filename:
                    lang = "zh"
                elif ".zh-Hans." in filename:
                    lang = "zh-Hans"
                elif ".zh-Hant." in filename:
                    lang = "zh-Hant"
                elif ".yue." in filename:
                    lang = "yue"

                try:
                    filepath = os.path.join(temp_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    parsed_content = self._parse_subtitle_text(content, "vtt")
                    if parsed_content.strip():
                        subtitles[lang] = parsed_content
                        print(f"✓ 从文件读取字幕: {lang}")

                except Exception as e:
                    print(f"读取字幕文件失败 {filename}: {str(e)}")

        return subtitles

    def extract_subtitles_with_transcript_api(self, video_id: str) -> Dict[str, str]:
        """
        使用youtube-transcript-api提取字幕 - 简单高效的方法

        Args:
            video_id: YouTube视频ID

        Returns:
            语言代码到字幕文本的映射
        """
        subtitles = {}

        try:
            # 创建API实例
            ytt_api = YouTubeTranscriptApi()

            # 优先语言列表
            preferred_langs = ["zh-HK", "zh-CN", "zh-TW", "zh", "yue", "en"]

            # 使用fetch方法直接获取字幕
            transcript_data = ytt_api.fetch(video_id, languages=preferred_langs)

            # 转换为纯文本
            text_lines = []
            for entry in transcript_data:
                text_lines.append(entry["text"])

            text = " ".join(text_lines).strip()

            if text:
                # 获取使用的语言
                lang = (
                    transcript_data._language_code
                    if hasattr(transcript_data, "_language_code")
                    else "unknown"
                )
                subtitles[lang.lower()] = text
                print(f"✓ youtube-transcript-api成功提取字幕: {lang}")
                return subtitles

        except Exception as e:
            print(f"youtube-transcript-api提取失败: {str(e)}")

        return subtitles

    def extract_subtitles_with_pytube(self, video_url: str) -> Dict[str, str]:
        """
        使用pytube提取字幕 - 处理XML格式字幕

        Args:
            video_url: YouTube视频URL

        Returns:
            语言代码到字幕文本的映射
        """
        subtitles = {}

        if not PYTUBE_AVAILABLE:
            print("pytube不可用，跳过此方法")
            return subtitles

        try:
            # 创建YouTube对象
            yt = YouTube(video_url)

            # 获取字幕轨道
            caption_tracks = yt.captions

            # 优先语言列表
            preferred_langs = ["zh-HK", "zh-CN", "zh-TW", "zh", "en"]

            for lang in preferred_langs:
                try:
                    if lang in caption_tracks:
                        # 获取字幕XML
                        caption = caption_tracks[lang]

                        # 生成字幕XML内容
                        xml_content = caption.xml_captions

                        # 解析XML并提取文本
                        import xml.etree.ElementTree as ET

                        root = ET.fromstring(xml_content)

                        text_parts = []
                        for elem in root.iter():
                            if elem.text and elem.text.strip():
                                text_parts.append(elem.text.strip())

                        text = " ".join(text_parts).strip()

                        if text:
                            subtitles[lang.lower()] = text
                            print(f"✓ pytube成功提取字幕: {lang}")
                            return subtitles  # 返回第一个成功的字幕

                except Exception as e:
                    print(f"pytube获取 {lang} 字幕失败: {str(e)}")
                    continue

        except Exception as e:
            print(f"pytube字幕提取失败: {str(e)}")

        return subtitles

    def _read_local_subtitles(self, video_id: str) -> Dict[str, str]:
        """
        读取本地下载的字幕文件

        Args:
            video_id: 视频ID

        Returns:
            语言代码到字幕文本的映射
        """
        subtitles = {}
        subtitle_extensions = [
            ".zh-hk.vtt",
            ".zh.vtt",
            ".zh-Hans.vtt",
            ".zh-Hant.vtt",
            ".yue.vtt",
            ".en.vtt",
        ]

        for ext in subtitle_extensions:
            subtitle_file = os.path.join(self.output_dir, f"{video_id}{ext}")
            if os.path.exists(subtitle_file):
                try:
                    with open(subtitle_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # 解析字幕内容
                    parsed_content = self._parse_subtitle_text(content, "vtt")
                    if parsed_content.strip():
                        lang = ext.replace(".vtt", "").lstrip(".")
                        subtitles[lang] = parsed_content
                        print(f"✓ 读取本地字幕文件: {lang}")

                    # 删除临时字幕文件
                    os.remove(subtitle_file)

                except Exception as e:
                    print(f"读取字幕文件失败 {ext}: {str(e)}")

        return subtitles

    def batch_extract(
        self, urls: List[str], output_file: str = "youtube_data.json"
    ) -> List[Dict]:
        """
        批量提取多个YouTube视频的数据

        Args:
            urls: YouTube视频URL列表
            output_file: 输出JSON文件名

        Returns:
            提取的视频数据列表
        """
        results = []
        total_urls = len(urls)

        print(f"开始批量处理 {total_urls} 个YouTube视频...")

        output_path = os.path.join(self.output_dir, output_file)
        for i, url in tqdm(enumerate(urls, 1), total=total_urls):
            print(f"\n处理进度: {i}/{total_urls}")
            print(f"URL: {url}")

            video_data = self.extract_video_info(url)
            if video_data:
                results.append(video_data)

            # 添加延迟避免被限制
            if i < total_urls:
                time.sleep(self.delay)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n处理完成！成功提取 {len(results)}/{total_urls} 个视频")
        print(f"结果已保存到: {output_path}")

        return results

    def extract_from_file(self, input_file: str, output_file: str = None) -> List[Dict]:
        """
        从文件读取URL列表并批量提取

        Args:
            input_file: 包含YouTube URL的文件路径
            output_file: 输出文件名（可选）

        Returns:
            提取的视频数据列表
        """
        if not os.path.exists(input_file):
            print(f"输入文件不存在: {input_file}")
            return []

        urls = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line.startswith(("http://", "https://")):
                    urls.append(line)

        if not urls:
            print("文件中没有找到有效的YouTube URL")
            return []

        if not output_file:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"{base_name}_extracted.json"

        return self.batch_extract(urls, output_file)

    def get_subtitle_text(
        self, video_data: Dict, preferred_langs: List[str] = None
    ) -> Tuple[str, str]:
        """
        从视频数据中获取字幕文本

        Args:
            video_data: 视频数据字典
            preferred_langs: 优先语言列表

        Returns:
            (字幕文本, 使用的语言代码)
        """
        if not preferred_langs:
            preferred_langs = ["zh-hk", "zh", "zh-Hans", "zh-Hant", "en"]

        subtitles = video_data.get("subtitles", {})

        for lang in preferred_langs:
            if lang in subtitles and subtitles[lang].strip():
                return subtitles[lang], lang

        # 如果没有找到首选语言，返回第一个可用的字幕
        for lang, text in subtitles.items():
            if text.strip():
                return text, lang

        return "", ""

    def extract_channel_urls(self, channel_url: str) -> List[str]:
        """
        从YouTube频道URL中提取所有视频URL

        Args:
            channel_url: YouTube频道URL (@ChannelName 或 channel/ChannelID)

        Returns:
            频道中所有视频的URL列表
        """
        # 标准化频道URL格式
        if channel_url.startswith("https://www.youtube.com/@"):
            # @格式的频道URL
            pass
        elif channel_url.startswith("https://www.youtube.com/channel/"):
            # 传统channel ID格式
            pass
        elif channel_url.startswith("https://www.youtube.com/c/"):
            # 自定义URL格式
            pass
        else:
            # 如果不是完整的URL，尝试构造
            if not channel_url.startswith("https://"):
                if channel_url.startswith("@"):
                    channel_url = f"https://www.youtube.com/{channel_url}/videos"
                else:
                    channel_url = (
                        f"https://www.youtube.com/channel/{channel_url}/videos"
                    )

        print(f"处理频道URL: {channel_url}")

        playlist_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,  # 只提取元数据，不下载
            "skip_download": True,
        }

        video_urls = []

        try:
            with yt_dlp.YoutubeDL(playlist_opts) as ydl:
                # 提取频道信息
                channel_info = ydl.extract_info(channel_url, download=False)

                if not channel_info:
                    print(f"无法获取频道信息: {channel_url}")
                    return []

                # 检查是否有视频列表
                if "entries" in channel_info:
                    entries = channel_info["entries"]
                    if entries:
                        print(
                            f"找到频道 '{channel_info.get('title', 'Unknown')}'，包含 {len(entries)} 个视频"
                        )

                        for entry in entries:
                            if entry and entry.get("id"):
                                video_id = entry["id"]
                                video_url = (
                                    f"https://www.youtube.com/watch?v={video_id}"
                                )
                                video_urls.append(video_url)
                                print(
                                    f"  - {entry.get('title', 'Unknown Title')} ({video_id})"
                                )
                    else:
                        print("频道为空")
                        return []

                else:
                    print(f"无法从URL提取频道视频: {channel_url}")
                    return []

        except Exception as e:
            print(f"提取频道失败: {channel_url} - {str(e)}")
            return []

        print(f"成功提取 {len(video_urls)} 个视频URL")
        return video_urls

    def extract_playlist_urls(self, playlist_url: str) -> List[str]:
        """
        从YouTube播放列表URL中提取所有视频URL

        Args:
            playlist_url: YouTube播放列表URL

        Returns:
            播放列表中所有视频的URL列表
        """
        import urllib.parse

        # 解析URL，提取播放列表ID
        parsed_url = urllib.parse.urlparse(playlist_url)
        query_params = urllib.parse.parse_qs(parsed_url.query)

        playlist_id = query_params.get("list", [None])[0]
        if not playlist_id:
            print(f"URL中没有找到播放列表ID: {playlist_url}")
            return []

        # 构造标准的播放列表URL
        standard_playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
        print(f"提取播放列表ID: {playlist_id}")
        print(f"使用标准播放列表URL: {standard_playlist_url}")

        playlist_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,  # 只提取元数据，不下载
            "skip_download": True,
        }

        video_urls = []

        try:
            with yt_dlp.YoutubeDL(playlist_opts) as ydl:
                # 提取播放列表信息
                playlist_info = ydl.extract_info(standard_playlist_url, download=False)

                if not playlist_info:
                    print(f"无法获取播放列表信息: {standard_playlist_url}")
                    return []

                # 检查是否是播放列表
                if "entries" in playlist_info:
                    entries = playlist_info["entries"]
                    if entries:
                        print(
                            f"找到播放列表 '{playlist_info.get('title', 'Unknown')}'，包含 {len(entries)} 个视频"
                        )

                        for entry in entries:
                            if entry and entry.get("id"):
                                video_id = entry["id"]
                                video_url = (
                                    f"https://www.youtube.com/watch?v={video_id}"
                                )
                                video_urls.append(video_url)
                                print(
                                    f"  - {entry.get('title', 'Unknown Title')} ({video_id})"
                                )
                    else:
                        print("播放列表为空")
                        return []

                else:
                    print(f"URL不是有效的播放列表: {standard_playlist_url}")
                    return []

        except Exception as e:
            print(f"提取播放列表失败: {standard_playlist_url} - {str(e)}")
            return []

        print(f"成功提取 {len(video_urls)} 个视频URL")
        return video_urls

    def extract_channel(self, channel_url: str, output_file: str = None) -> List[Dict]:
        """
        提取频道中所有视频的数据

        Args:
            channel_url: YouTube频道URL
            output_file: 输出文件名（可选）

        Returns:
            提取的所有视频数据列表
        """
        # 提取频道中的所有视频URL
        video_urls = self.extract_channel_urls(channel_url)

        if not video_urls:
            print("频道为空或提取失败")
            return []

        # 如果没有指定输出文件名，使用频道名称作为文件名
        if not output_file:
            # 从URL中提取频道名称
            if "@" in channel_url:
                channel_name = channel_url.split("@")[1].split("/")[0]
            else:
                channel_name = "unknown_channel"
            output_file = f"channel_{channel_name}.json"

        print(f"开始处理频道中的 {len(video_urls)} 个视频...")

        # 批量提取视频数据
        return self.batch_extract(video_urls, output_file)

    def extract_playlist(
        self, playlist_url: str, output_file: str = None
    ) -> List[Dict]:
        """
        提取播放列表中所有视频的数据

        Args:
            playlist_url: YouTube播放列表URL
            output_file: 输出文件名（可选）

        Returns:
            提取的所有视频数据列表
        """
        # 提取播放列表中的所有视频URL
        video_urls = self.extract_playlist_urls(playlist_url)

        if not video_urls:
            print("播放列表为空或提取失败")
            return []

        # 如果没有指定输出文件名，使用播放列表ID作为文件名
        if not output_file:
            # 从URL中提取播放列表ID
            import urllib.parse

            parsed_url = urllib.parse.urlparse(playlist_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            playlist_id = query_params.get("list", ["unknown"])[0]
            output_file = f"playlist_{playlist_id}.json"

        print(f"开始处理播放列表中的 {len(video_urls)} 个视频...")

        # 批量提取视频数据
        return self.batch_extract(video_urls, output_file)


def transcribe_audio(audio_url: str) -> str:
    print(f"Transcribing audio: {audio_url}")
    messages = [
        {
            "role": "system",
            "content": [
                {"text": "请识别音频中的粤语内容，并返回粤语文本。"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"audio": audio_url},
            ],
        },
    ]

    api_key = os.getenv("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    dashscope.api_key = api_key

    response = dashscope.MultiModalConversation.call(
        model="qwen3-asr-flash",
        messages=messages,
        result_format="message",
        asr_options={
            "language": "yue",  # 可选，若已知音频的语种，可通过该参数指定待识别语种，以提升识别准确率
            "enable_lid": True,
            "enable_itn": False,
        },
    )
    print(response.output.choices[0].message.content[0]["text"])
    return response.output.choices[0].message.content[0]["text"]


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="YouTube视频批量数据抓取工具")
    parser.add_argument("--urls", nargs="+", help="YouTube视频URL列表")
    parser.add_argument("--input-file", help="包含URL列表的输入文件")
    parser.add_argument("--playlist", help="YouTube播放列表URL，提取该列表中的所有视频")
    parser.add_argument("--channel", help="YouTube频道URL，提取该频道中的所有视频")
    parser.add_argument("--output-file", default="youtube_data.json", help="输出JSON文件名")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--retry", type=int, default=3, help="重试次数")
    parser.add_argument("--delay", type=float, default=1.0, help="请求间隔延迟（秒）")
    parser.add_argument("--max-duration", type=int, default=30, help="最大视频时长（秒）")
    parser.add_argument(
        "--subtitle-delay", type=float, default=0.5, help="字幕下载延迟倍数（相对于主延迟）"
    )
    parser.add_argument("--skip-subtitles", action="store_true", help="跳过字幕下载，只获取视频信息")
    parser.add_argument("--use-proxy", action="store_true", help="使用代理服务器")
    parser.add_argument(
        "--proxy-url", help="代理服务器URL (例如: http://proxy.example.com:8080)"
    )
    args = parser.parse_args()

    # 创建提取器
    extractor = YouTubeExtractor(
        output_dir=args.output_dir,
        retry_times=args.retry,
        delay=args.delay,
        subtitle_delay=args.subtitle_delay,
        skip_subtitles=args.skip_subtitles,
        use_proxy=args.use_proxy,
        proxy_url=args.proxy_url,
    )

    # 处理输入
    if args.channel:
        extractor.extract_channel(args.channel, args.output_file)
    elif args.playlist:
        extractor.extract_playlist(args.playlist, args.output_file)
    elif args.urls:
        extractor.batch_extract(args.urls, args.output_file)
    elif args.input_file:
        extractor.extract_from_file(args.input_file, args.output_file)
    else:
        print("请提供URL列表 (--urls)、输入文件 (--input-file)、播放列表 (--playlist) 或频道 (--channel)")
        print("使用示例:")
        print(
            "  python youtube_extract.py --urls 'https://www.youtube.com/watch?v=VIDEO_ID1' 'https://www.youtube.com/watch?v=VIDEO_ID2'"
        )
        print("  python youtube_extract.py --input-file urls.txt")
        print(
            "  python youtube_extract.py --playlist 'https://www.youtube.com/playlist?list=PLAYLIST_ID'"
        )
        print(
            "  python youtube_extract.py --channel 'https://www.youtube.com/@ChannelName/videos'"
        )


if __name__ == "__main__":
    # 新的流程：使用 yt-dlp 替代 pytube 获取音频，防止 HTTP 400 错误
    data = json.load(open("C4Styli/slogans/youtube_data.json", "r"))
    position = False
    results = []
    for item in tqdm(data, desc="Processing videos"):
        title = item["title"]
        youtube_url = item["url"]
        if not position:
            if youtube_url == "https://www.youtube.com/watch?v=xpJgtM11jAM":
                position = True
            continue
        audio_output = f"audio/{youtube_url.split('=')[-1]}_audio"

        if not os.path.exists(audio_output):
            if not os.path.exists("output"):
                os.makedirs("output")
            ytdlp_opts = {
                "format": "bestaudio/best",
                "outtmpl": audio_output,
                "quiet": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            }
            with yt_dlp.YoutubeDL(ytdlp_opts) as ydl:
                ydl.download([youtube_url])
            print("音频已下载:", audio_output)
        else:
            print("音频已存在:", audio_output)

        response = transcribe_audio(audio_url=audio_output + ".mp3")
        results.append(
            {
                "title": title,
                "url": youtube_url,
                "transcript": response,
            }
        )
        json.dump(
            results,
            open("C4Styli/slogans/youtube_slogans (probe).json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        os.remove(audio_output + ".mp3")
