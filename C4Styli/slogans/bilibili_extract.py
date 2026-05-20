import json
import os
import re
import time
from typing import List, Optional

import requests


def extract_bvid_from_url(url):
    """
    从B站视频URL中提取BVID

    Args:
        url: B站视频URL

    Returns:
        BVID字符串，如果未找到返回None
    """
    # 正则表达式匹配BV开头的ID
    pattern = r"BV[a-zA-Z0-9]+"
    match = re.search(pattern, url)

    if match:
        return match.group()
    else:
        return None


def extract_bvid_from_urls():
    # 测试URL
    test_urls = [
        "https://www.bilibili.com/video/BV1PU4y1c76R?spm_id_from=333.788.videopod.episodes&vd_source=9a3f8a782208921853d72ef5317a906d&p=137",
        "https://www.bilibili.com/video/BV1s7UrYpEQq/",
        "https://www.bilibili.com/video/BV1XVUiYSEMb/?p=1",
        "//www.bilibili.com/video/BV1FviyYsEFf/?spm_id_from=333.1387.collection.video_card.click",
    ]

    print("从B站视频URL中提取BVID:")
    print("=" * 50)

    for url in test_urls:
        bvid = extract_bvid_from_url(url)
        print(f"URL: {url}")
        print(f"BVID: {bvid}")
        print("-" * 30)

    # 如果用户想批量处理，可以从文件读取
    try:
        with open("extracted_urls.txt", "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        print(f"\n从文件处理 {len(urls)} 个URL:")

        bvids = []
        for url in urls:
            bvid = extract_bvid_from_url(url)
            if bvid:
                bvids.append(bvid)
                print(f"✓ {bvid}")
            else:
                print(f"✗ 无法提取: {url}")

        # 保存BVID到文件
        with open("batch_extracted_bvids.txt", "w", encoding="utf-8") as f:
            for bvid in bvids:
                f.write(bvid + "\n")

        print(f"\n成功提取 {len(bvids)} 个BVID，已保存到 batch_extracted_bvids.txt")

    except FileNotFoundError:
        print("\nextracted_urls.txt 文件不存在，跳过批量处理")


class BilibiliInfoExtractor:
    """Bilibili视频信息（包括title和字幕）提取器（支持批量获取UP主视频）"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            # 此cookie需替换为你自己的账户cookie，用于避免接口频率限制及部分B站接口需要权限
            "cookie": "SESSDATA=db63b3b2%2C1782579495%2C2bbbd%2Ac1CjAxZnXf94NADfbW1fciT-FW_LW1C2LvrkAeswsYTl6A2QUBOj53hECjrSsp3pILsdMSVmNmWHZSa3NnYjJETGJZZUV2TGV4Yk4wRVQzeHVvR0pHT1YyZWN3U0hTQlZzc0Y1QkhUOUppUzVqNElzeXAwSG9PbmVwNWVhXzFaWnhBMlY1cUFOb0p3IIEC",
        }

    def fetch_up_videos(
        self, mid: str, max_pages: int = 50, sleep: float = 2.0
    ) -> List[dict]:
        """
        批量提取UP主投稿视频BV号和基本信息。
        使用WBI签名接口获取更完整的视频信息
        mid: UP主uid
        max_pages: 最多抓取多少页（每页30个视频）
        sleep: 每页间隔秒数
        """
        all_videos = []
        page = 1
        page_size = 30

        while page <= max_pages:
            # 使用WBI签名接口，按发布时间排序
            url = f"https://api.bilibili.com/x/space/wbi/arc/search?mid={mid}&ps={page_size}&pn={page}&order=pubdate"

            try:
                resp = requests.get(url, headers=self.headers, timeout=15)
                resp.raise_for_status()
                j = resp.json()

                # Check for API-level errors
                if j.get("code") != 0:
                    print(
                        f"API Error: {j.get('message', 'Unknown error')} (code: {j.get('code')})"
                    )
                    # For rate limiting, wait longer and retry
                    if j.get("code") == -799:
                        print("Rate limited, waiting 5 seconds before retry...")
                        time.sleep(5)
                        continue
                    break

                # 获取视频列表
                vlist = j.get("data", {}).get("list", {}).get("vlist", [])
                if not vlist:
                    print(f"第{page}页没有更多视频")
                    break

                for v in vlist:
                    video_info = {
                        "bvid": v.get("bvid"),
                        "aid": v.get("aid"),
                        "title": v.get("title"),
                        "desc": v.get("description", ""),
                        "created": v.get("created"),
                        "video_review": v.get("video_review"),
                        "length": v.get("length"),
                        "play": v.get("play"),
                        "pic": v.get("pic"),
                        "comment": v.get("comment"),
                        "author": v.get("author"),
                    }
                    all_videos.append(video_info)

                print(f"第{page}页获取到{len(vlist)}个视频 (累计: {len(all_videos)}个)")

                # 如果这一页视频数量少于page_size，说明已经是最后一页
                if len(vlist) < page_size:
                    break

                page += 1
                time.sleep(sleep)

            except Exception as e:
                print(f"抓取UP主视频列表第{page}页失败: {e}")
                break

        return all_videos

    def extract_aid_cid_title(
        self, bvid: str, p: str = "1"
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        提取视频的aid、cid和标题title（主标题）
        """
        try:
            url = f"https://www.bilibili.com/video/{bvid}/?p={p}"
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            text = resp.text

            # aid
            aid_match = re.search(r'"aid[":\s]+(\d+)', text)
            aid = aid_match.group(1) if aid_match else None

            if not aid:
                aid_match = re.search(r"aid=(\d+)", text)
                aid = aid_match.group(1) if aid_match else None

            # cid
            cid_match = re.search(r'"cid[":\s]+(\d+)', text)
            cid = cid_match.group(1) if cid_match else None

            if not cid:
                cid_match = re.search(r"cid=(\d+)", text)
                cid = cid_match.group(1) if cid_match else None

            # title （兼容新版、旧版页面）
            title_match = re.search(r"<title.*?>(.*?)</title>", text, re.S)
            title = title_match.group(1).strip() if title_match else None
            if title:
                title = re.sub(r"_P?\d+.*?哔哩哔哩[_ ]?bilibili.*$", "", title)
                title = re.sub(r"_哔哩哔哩.*$", "", title)
                title = title.strip(" _-")

            # meta备用
            if not title:
                meta_title = re.search(
                    r'<meta\s+name="title"\s+content="([^"]+)"', text
                )
                if meta_title:
                    title = meta_title.group(1).strip()

            return aid, cid, title
        except Exception as e:
            print(f"提取aid/cid/title失败({bvid}): {e}")
            return None, None, None

    def get_subtitle_url(self, aid: str, cid: str) -> Optional[str]:
        """获取字幕URL"""
        try:
            api_url = f"https://api.bilibili.com/x/player/wbi/v2?aid={aid}&cid={cid}"
            response = requests.get(api_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            response_json = response.json()
            if response_json.get("code") != 0:
                return None, response_json

            subtitle_info = response_json.get("data", {}).get("subtitle", {})
            subtitles = subtitle_info.get("subtitles", [])

            if not subtitles:
                return None, response_json

            first_subtitle = subtitles[0]
            subtitle_url = first_subtitle.get("subtitle_url")
            if not subtitle_url:
                return None, response_json
            return (
                f"https:{subtitle_url}"
                if subtitle_url.startswith("//")
                else subtitle_url
            ), response_json
        except Exception as e:
            return None, response_json

    def extract_subtitles(self, subtitle_url: str) -> List[str]:
        """提取字幕内容（如有）"""
        try:
            response = requests.get(subtitle_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "body" not in data:
                return []
            contents = [
                item.get("content", "") for item in data["body"] if "content" in item
            ]
            return contents
        except Exception as e:
            return []

    def get_video_slogan_info(self, bvid: str) -> dict:
        """
        获取一个视频的title和（如有）全部第一P字幕
        """
        info = {"bvid": bvid}
        aid, cid, title = self.extract_aid_cid_title(bvid, "1")
        info.update({"aid": aid, "cid": cid, "title": title})
        if aid and cid:
            subtitle_url, response_json = self.get_subtitle_url(aid, cid)
            if subtitle_url:
                subtitles = self.extract_subtitles(subtitle_url)
                info["subtitles"] = subtitles
        return info, response_json


def main():
    out_json = "C4Styli/slogans/bilibili_slogans (probe).json"
    extractor = BilibiliInfoExtractor()
    # with open("C4Styli/slogans/bilibili_vidlist.txt", "r", encoding="utf-8") as f:
    #     videos = [line.strip() for line in f]

    videos = [
        "BV1Cx411k7Tw",
        "BV1Gt4y1k7Ws",
        "BV1u4411J7S8",
        "BV1Sh411y7Mc",
        "BV1Sp411d7H5",
        "BV1b91oYAEEY",
    ]

    if os.path.exists(out_json):
        out_data = json.load(open(out_json, "r", encoding="utf-8"))

    else:
        out_data = []

    searched_bvid = [item["metadata"]["bvid"] for item in out_data]

    for idx, bvid in enumerate(videos, 1):
        print(f"Processing {idx}/{len(videos)}")
        if bvid in searched_bvid:
            continue
        slogan_info, response_json = extractor.get_video_slogan_info(bvid)
        # 追加基础视频结构字段
        out_data.append({"metadata": slogan_info, "response_json": response_json})

        # 每处理完一个视频就保存一次，避免丢失进度
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        # 可优化为断点续存，或加个sleep降低频率
        time.sleep(2.0)

        # # 限制处理数量用于测试，生产环境可以移除这个限制
        # if idx >= 20:  # 改为20个用于测试
        #     print("达到测试限制数量，停止处理。如需处理全部视频，请修改此限制。")
        #     break

    print(f"保存结果到 {out_json} ...")
    print("Done.")


if __name__ == "__main__":
    main()
