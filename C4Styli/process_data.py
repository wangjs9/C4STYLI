from hmac import new
import json
from math import e
import urllib.parse
import urllib.request
import time
import re

# 尝试导入可选依赖
try:
    import requests
    from bs4 import BeautifulSoup
    USE_ADVANCED_PARSING = True
    print("使用高级解析模式 (requests + beautifulsoup4)")
except ImportError:
    USE_ADVANCED_PARSING = False
    print("使用基础解析模式 (仅使用标准库)")
    print("提示: 安装 requests 和 beautifulsoup4 可获得更好的解析效果")
    print("安装命令: pip install requests beautifulsoup4")

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    print("tqdm 未安装，使用基础进度显示")

def search_bilibili_videos(keyword, max_results=5):
    """
    搜索 bilibili 视频
    返回包含关键词的视频链接列表
    实现直接解析 <script id="__NEXT_DATA__"> 的 json，从而高质量稳定抓取B站搜索列表。
    """
    try:
        encoded_keyword = urllib.parse.quote(keyword)
        url = f"https://search.bilibili.com/all?keyword={encoded_keyword}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # always use requests+bs4 for robust parsing
        if USE_ADVANCED_PARSING:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            html = response.text
            # 抓取主搜索页中的 __NEXT_DATA__，包含结构化结果
            m = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
            items = []
            if m:
                import json as _json
                data = _json.loads(m.group(1))
                # 兼容 bilibili 不同搜索接口的 JSON 结构
                try:
                    cards = (
                        data['props']['pageProps']['dehydratedState']['queries'][0]['state']['data']['result']
                    )
                    for card in cards:
                        if card.get('result_type') == 'video':
                            for v in card.get('data', []):
                                items.append({
                                    "title": v.get("title", ""),
                                    "url": f"https://www.bilibili.com/video/{v.get('bvid','')}",
                                    "platform": "bilibili"
                                })
                        elif card.get('result_type') == 'video_v2':
                            for v in card.get('data', []):
                                items.append({
                                    "title": v.get("title", ""),
                                    "url": f"https://www.bilibili.com/video/{v.get('bvid','')}",
                                    "platform": "bilibili"
                                })
                except Exception as _:
                    # fallback: 没找到新式json结构，尝试旧结构
                    pass
            # 过滤包含关键词的视频并按max_results截断
            keyword_in_title = keyword.lower().replace("经典广告：", "").replace("經典廣告：", "").strip()
            def match(t):
                # 有些title有b站的关键字高亮<b>标签，去标签
                plain = re.sub(r"<.*?>", "", t)
                return keyword_in_title in plain.lower()
            results = [i for i in items if match(i['title'])][:max_results]
            # 不用再check, 直接返回
            return results

        else:
            # fallback to正则抓（弱）
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                html_content = response.read().decode('utf-8')
            video_pattern = r'<a.*?class="img-anchor".*?href="(.*?)".*?><div.*?><img.*?alt="(.*?)"'
            matches = re.findall(video_pattern, html_content, re.DOTALL)
            results = []
            for href, alt in matches[:max_results]:
                video_url = 'https:' + href if href.startswith('//') else href
                title = re.sub(r"<.*?>", "", alt)
                if keyword.replace("经典广告：", "").replace("經典廣告：", "").strip().lower() in title.strip().lower():
                    results.append({
                        'title': title,
                        'url': video_url,
                        'platform': 'bilibili'
                    })
            return results

    except Exception as e:
        print(f"Bilibili 搜索失败: {e}")
        return []

def search_youtube_videos(keyword, max_results=5):
    """
    搜索 YouTube 视频
    返回包含关键词的视频链接列表
    新实现：直接在页面<script>里拿ytInitialData字段（json），官方结构分析。
    """
    try:
        import json as _json
        encoded_keyword = urllib.parse.quote(keyword)
        url = f"https://www.youtube.com/results?search_query={encoded_keyword}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        if USE_ADVANCED_PARSING:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            html = response.text
        else:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8')

        # 关键: 抓ytInitialData JSON结构体（这是真正的搜索结果）, 只抓第一个<script>ytInitialData
        m = re.search(r"ytInitialData\s*=\s*({.*?});", html, re.DOTALL)
        items = []
        if m:
            jsondata = _json.loads(m.group(1))
            # 依照YouTube页面实际结构挖出视频
            try:
                sects = jsondata['contents']['twoColumnSearchResultsRenderer']['primaryContents']['sectionListRenderer']['contents']
                for sect in sects:
                    if not isinstance(sect, dict):
                        continue
                    if 'itemSectionRenderer' in sect:
                        for v in sect['itemSectionRenderer']['contents']:
                            if "videoRenderer" in v:
                                vid = v["videoRenderer"]
                                title = ""
                                if "title" in vid and "runs" in vid["title"]:
                                    title = "".join([x["text"] for x in vid["title"]["runs"]])
                                url = "https://www.youtube.com/watch?v=" + vid.get("videoId", "")
                                items.append({
                                    "title": title,
                                    "url": url,
                                    "platform": "youtube"
                                })
                            if len(items) >= max_results:
                                break
                    if len(items) >= max_results:
                        break
            except Exception as _:
                pass
        # 按关键词过滤（去掉"经典广告："等修饰）
        keyword_in_title = keyword.lower().replace("经典广告：", "").replace("經典廣告：", "").strip()
        def match(t):
            return keyword_in_title in t.lower()
        results = [i for i in items if match(i['title'])]
        if not results:
            # fallback: 以前的正则（弱），极少命中
            video_pattern = r'href="(/watch\?v=[^"]+)"[^>]*>(.*?)</a>'
            matches = re.findall(video_pattern, html, re.DOTALL)
            for href, anchor_html in matches:
                title_match = re.search(r'title="([^"]+)"', anchor_html)
                if title_match:
                    title = title_match.group(1)
                else:
                    tag_text = re.sub(r'<.*?>', '', anchor_html).strip()
                    title = tag_text
                if keyword_in_title in title.lower():
                    results.append({
                        'title': title,
                        'url': f"https://www.youtube.com{href}",
                        'platform': 'youtube'
                    })
                if len(results) >= max_results:
                    break
        return results[:max_results]

    except Exception as e:
        print(f"YouTube 搜索失败: {e}")
        return []

def process_slogan():
    with open("C4Styli/slogans/advertise.json", "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    new_data = []
    
    if USE_TQDM:
        data_iter = tqdm(data, total=len(data), desc="Processing slogans")
    else:
        data_iter = data

    for line in data_iter:
        new_line = {"company": line["company"], "product": line.get("product", ""), "slogan": line["slogan"], "year": line.get("year", -1), "region": line["region"]}
        if line.get("year", -1) == -1:
            # 爬取相关视频
            video_results = []
            search_keywords = [
                f"{line['company']} {line['slogan']}",  # 公司名 + 标语
                line['company'],  # 公司名
                line['slogan']    # 标语
            ]

            for keyword in search_keywords:
                if line["region"] == "HK":
                    keyword = "經典廣告：" + keyword
                else:
                    keyword = "经典广告：" + keyword
                print(f"正在搜索: {keyword}")

                if USE_ADVANCED_PARSING:
                    bilibili_results = search_bilibili_videos(keyword, max_results=3)
                    bilibili_results = [res for res in bilibili_results if any(char.isdigit() for char in res.get('title', ''))]
                    video_results.extend(bilibili_results)

                    youtube_results = search_youtube_videos(keyword, max_results=3)
                    youtube_results = [res for res in youtube_results if any(char.isdigit() for char in res.get('title', ''))]
                    video_results.extend(youtube_results)

                    # 避免请求过于频繁
                    time.sleep(1)
                else:
                    print(f"  跳过网络搜索（需要安装 requests 和 beautifulsoup4）")
                    print(f"  手动搜索建议: 在 bilibili 和 YouTube 上搜索 '{keyword}'")

            # 去重并添加到数据中
            unique_videos = []
            seen_urls = set()
            for video in video_results:
                if video['url'] not in seen_urls:
                    unique_videos.append(video)
                    seen_urls.add(video['url'])

            new_line['videos'] = unique_videos

            if unique_videos:
                print(f"找到 {len(unique_videos)} 个相关视频")
                for video in unique_videos[:3]:  # 只显示前3个
                    print(f"  - {video['platform']}: {video['title']} ({video['url']})")
            else:
                print("未找到相关视频")
        new_data.append(new_line)
            
    with open("C4Styli/slogans/advertise.jsonl", "w") as f:
        f.writelines("\n".join([json.dumps(line, ensure_ascii=False) for line in new_data]))
    
    with open("C4Styli/slogans/advertise.json", "w", encoding="utf-8-sig") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    
        
def process_movie():
    pass

def main():
    data_path = ""


if __name__ == "__main__":
    import os
    print(os.path.abspath(os.path.curdir))
    datapath = "C4Styli/slogans/digitaling_data.json"
    data = json.load(open(datapath, "r"))
    new_data = []
    for item in data:
        title = item["title"]
        date = item["date"]
        company = title.split("：")[0].split("×")[0]
        new_item = {
            "company": company,
            "product": "",
            "slogan": title,
            "date": date,
            "region": "CN"
        }
        new_data.append(new_item)
    
    with open("C4Styli/slogans/digitaling_data(1).json", "w") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
            
