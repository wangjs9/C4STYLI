import json
import multiprocessing
import os
import re
import sys

import requests
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from tqdm import tqdm
from utils import *


def get_wiki_lang_links(page_url, lang_code_dict):
    """
    获取不同语言页面的链接。
    lang_code_dict: {'en': 'English', 'zh': 'Chinese', 'zh-yue': 'Hong Kong Chinese'}
    返回: {lang_code: url}
    """
    headers = {
        "User-Agent": "CultureAwarenessBot/1.0 (https://github.com/jiashuo/culture-awareness; jiashuo@example.com) Python-requests/2.25.1"
    }
    try:
        r = requests.get(page_url, headers=headers, timeout=10)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        links = {}

        # 获取多语言链接
        for li in soup.select("li.interlanguage-link"):
            a = li.find("a")
            if a and a.has_attr("lang"):
                code = a["lang"].lower()
                if code in lang_code_dict:
                    href = a["href"]
                    if href.startswith("//"):
                        links[code] = "https:" + href
                    elif href.startswith("/"):
                        links[code] = f"https://{code}.wikipedia.org{href}"
                    else:
                        links[code] = href

        # 备用方法：查找hreflang属性
        for a in soup.find_all("a", attrs={"hreflang": True}):
            code = a["hreflang"].lower()
            if code in lang_code_dict and code not in links:
                links[code] = (
                    "https:" + a["href"] if a["href"].startswith("//") else a["href"]
                )

        return links
    except Exception as e:
        print(f"Error getting language links for {page_url}: {e}")
        return {}


def get_plot_summary(page_url, require_plot_section=True):
    """
    获取维基百科页面的剧情简介部分。
    专门寻找Plot/Synopsis章节的内容。
    如果require_plot_section=True，则只返回有专门plot章节的页面内容
    """
    headers = {
        "User-Agent": "CultureAwarenessBot/1.0 (https://github.com/jiashuo/culture-awareness; jiashuo@example.com) Python-requests/2.25.1"
    }
    try:
        r = requests.get(page_url, headers=headers, timeout=10)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        if "/zh-hk/" in page_url:
            h1 = soup.find("h1", id="firstHeading")
            if (
                h1
                and h1.has_attr("lang")
                and h1["lang"].lower() in ("zh-hans-hk", "zh-hk")
            ):
                title = h1.get_text(strip=True)
            else:
                label = soup.find(
                    lambda tag: tag.name == "th"
                    and tag.get_text(strip=True) in ("中国香港", "香港")
                )
                if label:
                    td = label.find_next("td")
                    if td:
                        title = td.get_text(strip=True)
                if not title and h1:
                    title = h1.get_text(strip=True)  # fallback
        elif "/zh-cn/" in page_url:
            # 如果有 zh-cn，需要有 <h1 ... lang="zh-Hans-CN">
            h1 = soup.find("h1", id="firstHeading")
            if (
                h1
                and h1.has_attr("lang")
                and h1["lang"].lower() in ("zh-hans-cn", "zh-cn")
            ):
                title = h1.get_text(strip=True)
            else:
                label = soup.find(
                    lambda tag: tag.name == "th"
                    and tag.get_text(strip=True) in ("中国大陆", "大陆")
                )
                if label:
                    td = label.find_next("td")
                    if td:
                        title = td.get_text(strip=True)
                if not title and h1:
                    title = h1.get_text(strip=True)  # fallback

        else:
            # 不是地区特殊页面，直接抓
            h1 = soup.find("h1", id="firstHeading")
            title = h1.get_text(strip=True) if h1 else ""

        # 寻找剧情章节
        plot_text = ""
        has_plot_section = False

        # 查找Plot或Synopsis标题
        plot_headers = []
        for h2 in soup.find_all(["h2", "h3"]):
            # 尝试多种方式获取标题文本
            header_text = ""
            span = h2.find("span", class_="mw-headline")
            if span:
                header_text = span.get_text().strip().lower()
            else:
                header_text = h2.get_text().strip().lower()

            # 移除可能的[编辑]等额外文本
            header_text = header_text.split("[")[0].strip()

            # 根据页面URL判断语言，使用不同的关键词
            plot_keywords = []
            if "zh.wikipedia.org" in page_url or "zh-yue.wikipedia.org" in page_url:
                # 中文和粤语页面
                plot_keywords = [
                    "剧情",
                    "情節",
                    "劇情",
                    "故事",
                    "内容简介",
                    "故事概要",
                    "內容簡介",
                    "故事大綱",
                    "情节",
                ]
            else:
                # 英文页面
                plot_keywords = ["plot", "synopsis"]

            if any(keyword in header_text for keyword in plot_keywords):
                plot_headers.append(h2)

        if plot_headers:
            has_plot_section = True
            # 获取剧情章节的内容
            plot_header = plot_headers[0]
            content_parts = []

            # 从plot标题开始，向下查找所有段落，直到遇到下一个h2/h3标题
            for elem in plot_header.find_all_next():
                if elem.name in ["h2", "h3"] and elem != plot_header:
                    # 遇到下一个标题，停止
                    break
                if elem.name == "p":
                    text = elem.get_text(strip=True)
                    if (
                        text
                        and not text.startswith(("坐标", "Coordinates"))
                        and len(text) > 20
                    ):
                        content_parts.append(text)

            plot_text = " ".join(content_parts[:3])  # 取前3段作为剧情简介

        # 如果没找到专门的剧情章节
        if not plot_text:
            if require_plot_section:
                # 如果要求必须有plot章节，返回空
                return title, "", False
            else:
                # 回退到第一段
                for p in soup.select("div.mw-parser-output > p"):
                    text = p.get_text(strip=True)
                    if (
                        text
                        and not text.startswith(
                            ("坐标", "Coordinates", "Redirected from")
                        )
                        and len(text) > 50
                    ):
                        plot_text = text
                        break
                # 如果成功获取到回退内容，认为有plot（即使不是专门的章节）
                if plot_text:
                    has_plot_section = True

        return title, plot_text, has_plot_section
    except Exception as e:
        print(f"Error getting plot for {page_url}: {e}")
        return "", "", False


def process_single_movie(movie_info):
    """
    处理单个电影的数据获取任务，适合进程池使用
    返回电影数据字典或None（如果处理失败）
    """
    movie_title, en_url, year = movie_info

    # 固定的语言代码字典
    lang_codes = {
        "en": "English",
        "zh": "Chinese",
        "yue": "Yue",
    }

    try:
        # 检查语言链接，优先使用 zh-cn，如果没有则用 zh
        lang_links = get_wiki_lang_links(en_url, lang_codes)

        zh_link = None
        if "zh" in lang_links:
            zh_link = lang_links["zh"]
            try:
                idx = zh_link.index("/wiki/")
                base = zh_link[:idx]
                target_name = zh_link[idx + len("/wiki/") :]
                zh_link = f"{base}/zh-cn/{target_name}"
                hk_link = f"{base}/zh-hk/{target_name}"
            except Exception as e:
                print(f"Error parsing zh wiki url: {zh_link} - {e}")
                return None
        else:
            return None

        if "yue" not in lang_links:
            return None

        # 获取各语言版本的内容，plot章节可选
        title_en, summary_en, has_plot_en = get_plot_summary(
            en_url, require_plot_section=False
        )
        title_zh, summary_zh, has_plot_zh = get_plot_summary(
            zh_link, require_plot_section=False
        )
        title_hk, summary_hk, has_plot_hk = get_plot_summary(
            hk_link, require_plot_section=False
        )

        if not has_plot_en:
            return None
        if not has_plot_zh:
            summary_zh = ""
        if not has_plot_hk:
            summary_hk = ""

        cn_traditional = norm_text(title_zh, "HK")
        hk_simplified = norm_text(title_hk, "CN")

        if (
            cn_traditional == title_hk
            or title_zh == hk_simplified
            or title_zh == title_hk
        ):
            return None

        movie_data = {
            "YEAR": year,
            "TITLE": title_en,
            "TITLE (CN)": title_zh,
            "TITLE (HK)": title_hk,
            "PLOT SUMMARY": summary_en,
            "PLOT SUMMARY (CN)": summary_zh,
            "PLOT SUMMARY (HK)": summary_hk,
            "WIKI_EN": en_url,
            "WIKI_CN": zh_link,
            "WIKI_HK": hk_link,
        }

        return movie_data

    except Exception as e:
        print(f"Error processing movie {movie_title}: {e}")
        return None


def get_popular_english_movies():
    """
    获取热门英语电影列表（使用Wikipedia的特色电影列表）
    """
    headers = {
        "User-Agent": "CultureAwarenessBot/1.0 (https://github.com/jiashuo/culture-awareness; jiashuo@example.com) Python-requests/2.25.1"
    }

    # 使用Wikipedia的特色电影列表作为数据源
    urls = [
        "https://en.wikipedia.org/wiki/List_of_American_films_of_2023",
        "https://en.wikipedia.org/wiki/List_of_American_films_of_2022",
        "https://en.wikipedia.org/wiki/List_of_American_films_of_2021",
        "https://en.wikipedia.org/wiki/List_of_American_films_of_2020",
        "https://en.wikipedia.org/wiki/List_of_American_films_of_2019",
        "https://en.wikipedia.org/wiki/List_of_American_films_of_2024",
        "https://en.wikipedia.org/wiki/List_of_American_films_of_2025",
        "https://en.wikipedia.org/wiki/List_of_American_films_of_2018",
        "https://en.wikipedia.org/wiki/List_of_American_films_of_2017",
        "https://en.wikipedia.org/wiki/List_of_American_films_of_2016",
    ]

    movies = []
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")

            # 提取年份
            year_match = url.split("_")[-1]
            year = int(year_match) if year_match.isdigit() else 2023

            print(f"  处理年份 {year} 的电影列表...")

            # 查找电影表格
            tables = soup.find_all("table", class_="wikitable")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows[1:]:  # 跳过表头
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 2:
                        # 检查所有单元格中的电影链接（不只是第一列）
                        for cell in cells:
                            link = cell.find("a")
                            if (
                                link
                                and link.has_attr("href")
                                and "/wiki/" in link["href"]
                            ):
                                movie_title = link.get_text(strip=True)
                                movie_url = "https://en.wikipedia.org" + link["href"]

                                # 过滤明显不是电影的条目
                                if (
                                    len(movie_title) > 3
                                    and not any(
                                        skip in movie_title.lower()
                                        for skip in [
                                            "list",
                                            "category",
                                            "portal",
                                            "template",
                                            "films of",
                                            "see also",
                                        ]
                                    )
                                    and movie_title not in [m[0] for m in movies]
                                ):  # 避免重复
                                    movies.append((movie_title, movie_url, year))
                                    break  # 找到一个电影链接就停止检查这个行的其他单元格

            print(f"    从 {year} 年找到 {len([m for m in movies if m[2] == year])} 部电影")

        except Exception as e:
            print(f"Error getting movies from {url}: {e}")
            continue

    print(f"  总共找到 {len(movies)} 部电影")
    return movies


def fetch_english_movies_with_translations(max_movies=50000, num_processes=None):
    """
    爬取热门英语电影，要求同时有中文（简体 zh-cn）和粤语翻译版本。
    获取各语言版本的标题和剧情简介。
    使用多进程并发处理以提高速度。
    """
    movies = []

    print("获取热门英语电影列表...")
    all_movies = get_popular_english_movies()
    movies_to_process = all_movies[:max_movies]

    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), len(movies_to_process))

    print(f"开始多进程处理 {len(movies_to_process)} 部电影（使用 {num_processes} 个进程）...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(process_single_movie, movies_to_process)

        # 使用tqdm显示进度
        with tqdm(total=len(movies_to_process), desc="Processing movies") as pbar:
            for movie_data in results:
                if movie_data:
                    movies.append(movie_data)
                pbar.update(1)

    movies = [m for m in movies if m is not None]

    print(f"成功处理 {len(movies)} 部电影（总共尝试 {len(movies_to_process)} 部）")
    return movies


def wiki_movie_data():
    """
    爬取1900-2025年英语电影的题目和简介，以及对应的中文和粤语翻译版本。
    支持多进程并发抓取电影详情，提高抓取速度。
    主要加速点：同一批en电影链接下细粒度并发（每部电影的不同语言详情并发处理）
    """

    # 爬取电影数据
    print("开始爬取热门英语电影数据（要求同时有中文（简体）和粤语版本）...")
    print("将获取各语言版本的标题和剧情简介...")
    movies = fetch_english_movies_with_translations(max_movies=50000, num_processes=8)

    print(f"\n共收集到 {len(movies)} 部电影")

    # 显示前5个示例
    print("\n前5个电影示例：")
    for i, movie in enumerate(movies[:5]):
        print(f"{i+1}. {movie['TITLE']} ({movie['YEAR']})")
        print(f"   中文: {movie['TITLE (CN)']}")
        print(f"   粤语: {movie['TITLE (HK)']}")
        print(f"   英文剧情: {movie.get('PLOT SUMMARY','')[:100]}...")
        print(f"   中文剧情: {movie.get('PLOT SUMMARY (CN)','')[:100]}...")
        print(f"   粤语剧情: {movie.get('PLOT SUMMARY (HK)','')[:100]}...")
        print()

    with open("C4Styli/titles/drama_titles.json", "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)

    print("数据已保存到 drama_titles.json")

    return movies


# 辅助函数：用于处理drama数据（必须在模块级别以便multiprocessing序列化）
def _drama_extract_wiki_title_from_url(url):
    """从URL中提取wiki标题"""
    m = re.search(r"/wiki/([^/?#]+)", url)
    if m:
        return m.group(1)
    return None


def _drama_get_title_from_html(html):
    """从HTML中提取中文标题"""
    m = re.search(r"<title>(.*?) *- *维基百科，?自由(的)?百科全书</title>", html)
    if not m:
        # 再尝试香港繁体
        m = re.search(r"<title>(.*?) *- *維基百科，?自由.百科全書</title>", html)
    if m:
        return m.group(1).strip()
    return None


def _drama_get_plot_from_text(text):
    """从文本中提取剧情"""
    candidates = []
    # 剧情
    m = re.search(
        r"(剧情|情節|劇情|故事|内容简介|故事概要|內容簡介|故事大綱|情节|梗概)[\s]*\n([\s\S]{0,1500})", text
    )
    if m:
        # 提取后第一个<p>的段落
        para = re.search(r"[\n]*([\u4e00-\u9fa5][\s\S]{10,900})", m.group(2))
        if para:
            candidates.append(para.group(1).strip())
        else:
            candidates.append(m.group(2).strip())
    # 或都找所有 '剧情'/'故事'/'梗概'标题下第一个<p>
    for zh_label in ("剧情", "情節", "劇情", "故事", "内容简介", "故事概要", "內容簡介", "故事大綱", "情节"):
        # 寻找 == 剧情 == 或者二级标题
        m = re.search(r"={2,3}\s*" + zh_label + r"\s*={2,3}([\s\S]{0,900})", text)
        if m:
            para = re.search(r"[\n]*([\u4e00-\u9fa5][\s\S]{10,900})", m.group(1))
            if para:
                candidates.append(para.group(1).strip())
    # 任选有内容的第一个
    for c in candidates:
        if c and len(c) > 10:
            return c
    return ""


def _drama_extract_year_from_page(soup, title_text=""):
    """从维基百科页面提取年份"""
    year = None

    # 方法1: 从标题中提取 (YYYY) 或 (YYYY film) 格式
    if title_text:
        year_match = re.search(r"\((\d{4})", title_text)
        if year_match:
            year = int(year_match.group(1))
            # 确保年份合理（1900-2100）
            if 1900 <= year <= 2100:
                return year

    # 方法2: 从信息框(infobox)中提取
    infobox = soup.find("table", class_="infobox")
    if infobox:
        rows = infobox.find_all("tr")
        for row in rows:
            th = row.find("th")
            td = row.find("td")
            if th and td:
                label = th.get_text(strip=True).lower()
                value = td.get_text(strip=True)
                # 查找包含 release, date, year 等关键词的行
                if any(
                    keyword in label
                    for keyword in ["release", "date", "year", "aired", "premiere"]
                ):
                    # 尝试提取年份
                    year_match = re.search(r"\b(19|20)\d{2}\b", value)
                    if year_match:
                        candidate_year = int(year_match.group())
                        if 1900 <= candidate_year <= 2100:
                            year = candidate_year
                            return year

    # 方法3: 从第一段中提取年份（作为最后手段）
    first_p = soup.select_one("div.mw-parser-output > p")
    if first_p:
        text = first_p.get_text()
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        if year_match:
            candidate_year = int(year_match.group())
            if 1900 <= candidate_year <= 2100:
                year = candidate_year
                return year

    return year


def _process_one_drama(drama):
    """处理单个drama的数据获取任务，适合进程池使用"""
    headers = {
        "User-Agent": "CultureAwarenessBot/1.0 (https://github.com/jiashuo/culture-awareness; jiashuo@example.com) Python-requests/2.25.1"
    }

    drama_url = "https://en.wikipedia.org" + drama
    title_en, plot_en, has_plot_en = get_plot_summary(
        drama_url, require_plot_section=False
    )
    if not has_plot_en or not plot_en:
        return None

    # 提取年份
    year = None
    try:
        r = requests.get(drama_url, headers=headers, timeout=8)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        year = _drama_extract_year_from_page(soup, title_en)
    except Exception:
        pass

    drama_data = {
        "TITLE": title_en,
        "PLOT SUMMARY": plot_en,
    }
    if year:
        drama_data["YEAR"] = year
    else:
        drama_data["YEAR"] = -1

    # 使用 get_wiki_lang_links 获取正确的中文链接
    lang_codes = {
        "zh": "Chinese",
    }
    lang_links = get_wiki_lang_links(drama_url, lang_codes)

    zh_cn_url = None
    zh_hk_url = None

    if "zh" in lang_links:
        zh_base_url = lang_links["zh"]
        try:
            # 访问中文页面，从link标签中提取zh-Hans-CN和zh-Hant-HK链接
            zh_r = requests.get(zh_base_url, headers=headers, timeout=8)
            zh_r.encoding = zh_r.apparent_encoding
            zh_soup = BeautifulSoup(zh_r.text, "html.parser")

            # 从link标签中提取zh-Hans-CN和zh-Hant-HK
            zh_cn_link = zh_soup.find(
                "link", {"rel": "alternate", "hreflang": "zh-Hans-CN"}
            )
            zh_hk_link = zh_soup.find(
                "link", {"rel": "alternate", "hreflang": "zh-Hant-HK"}
            )

            if zh_cn_link:
                zh_cn_url = zh_cn_link.get("href", "")
                # 如果是相对路径，转换为绝对路径
                if zh_cn_url.startswith("//"):
                    zh_cn_url = "https:" + zh_cn_url
                elif zh_cn_url.startswith("/"):
                    zh_cn_url = "https://zh.wikipedia.org" + zh_cn_url

            if zh_hk_link:
                zh_hk_url = zh_hk_link.get("href", "")
                # 如果是相对路径，转换为绝对路径
                if zh_hk_url.startswith("//"):
                    zh_hk_url = "https:" + zh_hk_url
                elif zh_hk_url.startswith("/"):
                    zh_hk_url = "https://zh.wikipedia.org" + zh_hk_url

        except Exception as e:
            print(f"Error getting alternate links from {zh_base_url}: {e}")
            # 回退到原来的方法：从zh链接构建zh-cn和zh-hk
            try:
                idx = zh_base_url.index("/wiki/")
                base = zh_base_url[:idx]
                target_name = zh_base_url[idx + len("/wiki/") :]
                zh_cn_url = f"{base}/zh-cn/{target_name}"
                zh_hk_url = f"{base}/zh-hk/{target_name}"
            except Exception:
                pass

    # 如果无法获取语言链接，尝试使用英文标题构建（备用方案）
    if not zh_cn_url:
        drama_xxx = _drama_extract_wiki_title_from_url(drama_url)
        if drama_xxx:
            zh_cn_url = f"https://zh.wikipedia.org/zh-cn/{drama_xxx}"
            zh_hk_url = f"https://zh.wikipedia.org/zh-hk/{drama_xxx}"

    # 默认都用requests.get，加错误处理
    title_cn, plot_cn = "", ""
    if zh_cn_url:
        try:
            zh_cn_res = requests.get(zh_cn_url, headers=headers, timeout=8)
            zh_cn_res.encoding = zh_cn_res.apparent_encoding
            if zh_cn_res.status_code == 200:
                zh_cn_html = zh_cn_res.text
                title_cn = _drama_get_title_from_html(zh_cn_html) or ""
                plot_cn = _drama_get_plot_from_text(zh_cn_html)
        except Exception:
            title_cn, plot_cn = "", ""

    title_hk, plot_hk = "", ""
    if zh_hk_url:
        try:
            zh_hk_res = requests.get(zh_hk_url, headers=headers, timeout=8)
            zh_hk_res.encoding = zh_hk_res.apparent_encoding
            if zh_hk_res.status_code == 200:
                zh_hk_html = zh_hk_res.text
                title_hk = _drama_get_title_from_html(zh_hk_html) or ""
                plot_hk = _drama_get_plot_from_text(zh_hk_html)
        except Exception:
            title_hk, plot_hk = "", ""

    drama_data["TITLE (CN)"] = title_cn
    drama_data["PLOT SUMMARY (CN)"] = plot_cn
    drama_data["TITLE (HK)"] = title_hk
    drama_data["PLOT SUMMARY (HK)"] = plot_hk

    return drama_data


def wiki_drama_data():
    """
    爬取1900-2025年英语电视剧的题目和简介，以及对应的中文和粤语翻译版本。
    """
    with open("C4Styli/titles/drama_titles.txt", "r", encoding="utf-8") as f:
        drama_list = f.readlines()
    all_drama_data = []

    # 用多进程Pool批量处理，使用模块级别的_process_one_drama函数
    with multiprocessing.Pool(processes=8) as pool:
        results = list(
            tqdm(
                pool.imap(_process_one_drama, drama_list),
                total=len(drama_list),
                desc="Processing dramas (multiprocess)",
            )
        )

    all_drama_data.extend([r for r in results if r])

    with open("C4Styli/titles/drama_titles.json", "w", encoding="utf-8") as f:
        json.dump(all_drama_data, f, ensure_ascii=False, indent=2)

    print("电视剧数据已保存到 drama_titles.json")

    return all_drama_data


if __name__ == "__main__":
    # wiki_movie_data()
    wiki_drama_data()
