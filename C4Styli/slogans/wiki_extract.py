import json
import os
import re
import sys
import time
import unicodedata

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import requests
from bs4 import BeautifulSoup


def get_hkex_listed_companies(category_urls, max_pages=3, sleep_sec=1):
    """
    爬取维基百科Category页面，获取所有上市公司条目及其对应页面URL
    """
    companies = []
    base = "https://zh.wikipedia.org"
    for next_page in category_urls:
        page_cnt = 0
        while next_page and page_cnt < max_pages:
            res = requests.get(next_page, headers={"User-Agent": "Mozilla/5.0"})
            if not res.ok:
                print(f"请求失败: {res.status_code}")
                break
            soup = BeautifulSoup(res.text, "html.parser")
            mw_pages = soup.find("div", id="mw-pages")
            if not mw_pages:
                break
            links = mw_pages.find_all("a")
            for link in links:
                # 排除更多下一页链接
                href = link.get("href", "")
                title = link.get_text()
                if href.startswith("/wiki/") and not href.startswith("/wiki/Category:"):
                    companies.append(
                        {"company": title.strip(), "wiki_url": base + href}
                    )
            # 查找“下一页”链接
            next_div = mw_pages.find("a", string="下一页")
            if next_div:
                next_page = base + next_div["href"]
            else:
                next_page = None
            page_cnt += 1
            time.sleep(sleep_sec)
        print(f"获取上市公司条目: {len(companies)}")
    return companies


def extract_slogan_from_article(wiki_url, sleep_sec=0.5):
    """
    爬取单个公司维基百科条目，抓取公司名称/产品/中国大陆还是香港的公司，用过的标语口号和口号年限（可能有多个）
    返回 {
        "company": ...,
        "product": ...,
        "region": ...,  # "CN" 或 "HK" 或 "TW"等
        "slogans": [{"slogan":..., "year":...}, ...]
    }
    """

    def clean_text(s):
        return unicodedata.normalize("NFKC", s.strip().replace("\n", " "))

    def extract_html_slogan_field(td_html):
        """从含有html的soup td里，保留显著文本内容，去除sup脚注、span等无关内容。"""
        for tag in td_html.find_all(["sup", "span", "style", "script"]):
            tag.decompose()
        # 提取 <br> 为换行再拼
        texts = []
        for elem in td_html.children:
            if hasattr(elem, "name") and elem.name in ["br"]:
                texts.append("\n")
            elif isinstance(elem, str):
                texts.append(elem)
            else:
                texts.append(elem.get_text(separator="\n", strip=True))
        result = "\n".join([t.strip() for t in "".join(texts).split("\n") if t.strip()])
        return result

    time.sleep(sleep_sec)
    slogan_patterns = [r"标语口号[\s:：]*[「“\"']?([^」”\"'\n;。；，,]+)[」”\"']?"]
    year_patterns = [r"([12][09]\d{2})年"]
    region_map = {
        "中国大陆": "CN",
        "中国": "CN",
        "香港": "HK",
        "台灣": "TW",
        "台湾": "TW",
        "澳门": "MO",
        "澳門": "MO",
    }

    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(wiki_url, headers=headers)
    if not res.ok:
        return {}

    soup = BeautifulSoup(res.text, "html.parser")

    # 检查所有infobox
    infoboxes = soup.find_all("table", class_=re.compile("infobox"))

    company = None
    product = None
    region = None

    # 提取公司名、产品、地区
    for infobox in infoboxes:
        rows = infobox.find_all("tr")
        for row in rows:
            th = row.find("th")
            td = row.find("td")
            if not th or not td:
                continue
            k = th.get_text().strip()
            v = td.get_text().strip()
            if not company and any(word in k for word in ["公司名称", "企业名称", "名称", "中文名"]):
                company = clean_text(v)
            if not product and any(
                word in k for word in ["产品", "產品", "主要产品", "主要產品", "产业", "産業"]
            ):
                product = clean_text(v)
            if not region and any(word in k for word in ["总部地点", "国家", "地区", "区域"]):
                for key, val in region_map.items():
                    if key in v:
                        region = val
                        break

    if not company:
        title = soup.find("h1")
        company = clean_text(title.get_text()) if title else ""

    # region 必须抓出，找不到也返回None
    if not region:
        body_text = soup.get_text(separator="\n")
        for k, v in region_map.items():
            if k in body_text:
                region = v
                break
        # 保证有region字段，即使没抓到
        if not region:
            region = None

    slogans_found = []
    slogan_labels = ["标语口号", "標語口號", "標語", "口号", "口號", "广告语", "廣告語"]

    for infobox in infoboxes:
        for row in infobox.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td:
                th_text = th.get_text().strip()
                if any(lbl in th_text for lbl in slogan_labels):
                    content = extract_html_slogan_field(td)

                    added = set()
                    # 直接按行分割内容，处理每个标语
                    lines = [x.strip() for x in content.split("\n") if x.strip()]
                    current_slogan = None
                    current_year = None

                    for ln in lines:
                        # 跳过英语翻译行
                        if any(
                            x in ln for x in ["英语：", "英文：", "English:", "英语:", "英文:"]
                        ):
                            continue

                        # 检查是否是年份行，如"(2011年~)"
                        year_match = re.search(r"[（(](\d{4})年[~～]?\s*[)）]", ln)
                        if year_match:
                            if current_slogan is not None:
                                current_year = int(year_match.group(1))
                            continue

                        # 如果是中文标语行
                        if ln and not ln.startswith("（") and not ln.startswith("("):
                            # 保存之前的标语（如果有）
                            if current_slogan is not None:
                                slogans_found.append(
                                    {
                                        "slogan": current_slogan,
                                        "year": current_year
                                        if current_year is not None
                                        else -1,
                                    }
                                )
                                # 不再用 added 集合，允许重复出现

                            # 开始新标语
                            current_slogan = clean_text(ln)
                            current_year = None

                    # 保存最后一个标语
                    if current_slogan is not None:
                        slogans_found.append(
                            {
                                "slogan": current_slogan,
                                "year": current_year
                                if current_year is not None
                                else -1,
                            }
                        )

    # 若找不到，降级为全页文本扫描
    if not slogans_found:
        text2 = soup.get_text(separator="\n", strip=True)
        added2 = set()
        for pat in slogan_patterns:
            for m in re.finditer(pat, text2):
                slogan_raw = clean_text(m.group(1).replace("\n", " "))
                if slogan_raw and slogan_raw not in added2:
                    start_pos = m.start()
                    before = text2[max(0, start_pos - 50) : start_pos]
                    after = text2[m.end() : m.end() + 50]
                    year = None
                    for ypat in year_patterns:
                        yy1 = re.search(ypat, before)
                        yy2 = re.search(ypat, after)
                        if yy1:
                            year = int(yy1.group(1))
                            break
                        elif yy2:
                            year = int(yy2.group(1))
                            break
                    slogans_found.append({"slogan": slogan_raw, "year": year})
                    added2.add(slogan_raw)

    # 最后，再次降级：如果有infobox，直接取含标语字段但没被正则匹配到的内容
    if not slogans_found and infoboxes:
        for infobox in infoboxes:
            for row in infobox.find_all("tr"):
                th = row.find("th")
                td = row.find("td")
                if th and td:
                    if (
                        "口号" in th.get_text()
                        or "廣告語" in th.get_text()
                        or "广告语" in th.get_text()
                        or "標語" in th.get_text()
                    ):
                        val = extract_html_slogan_field(td)
                        if val and not any(s["slogan"] == val for s in slogans_found):
                            slogans_found.append(
                                {"slogan": clean_text(val), "year": None}
                            )

    if slogans_found:
        for s in slogans_found:
            if s["year"] is None:
                s["year"] = -1

    return {
        "company": company,
        "product": product,
        "region": region,
        "slogans": slogans_found,
    }


def crawl_hkex_company_slogans(
    category_url, out_json="hkex_company_slogans.json", limit=800
):
    """
    主函数：从HKEX公司category页面，批量爬取公司口号
    """
    companies = get_hkex_listed_companies(category_url)
    results = []
    cnt = 0
    for info in companies:
        if cnt >= limit:
            break
        print(f"{cnt+1}/{limit} {info['company']}")
        extra = extract_slogan_from_article(info["wiki_url"])

        slogans = extra.get("slogans", [])
        for slogan in slogans:
            result = {
                "company": info["company"],
                "wiki_url": info["wiki_url"],
                "product": extra.get("product", ""),
                "slogan": slogan["slogan"],
                "year": slogan["year"],
                "region": extra.get("region"),
            }
            results.append(result)
        cnt += 1
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"完成，写入{out_json}，共{len(results)}条。")


def main():
    hkex_cat_url = [
        "https://zh.wikipedia.org/wiki/Category:%E9%A6%99%E6%B8%AF%E6%99%82%E8%A3%9D%E5%93%81%E7%89%8C",
        "https://zh.wikipedia.org/wiki/Category:%E9%A6%99%E6%B8%AF%E9%9B%B6%E5%94%AE%E5%95%86",
        "https://zh.wikipedia.org/wiki/Category:%E9%A6%99%E6%B8%AF%E9%80%A3%E9%8E%96%E5%BA%97"
        "https://zh.wikipedia.org/wiki/Category:%E9%A6%99%E6%B8%AF%E6%9C%8D%E8%A3%85%E5%85%AC%E5%8F%B8",
        "https://zh.wikipedia.org/wiki/Category:%E9%A6%99%E6%B8%AF%E4%B8%8A%E5%B8%82%E7%B6%9C%E5%90%88%E4%BC%81%E6%A5%AD%E5%85%AC%E5%8F%B8",
    ]
    crawl_hkex_company_slogans(hkex_cat_url)


if __name__ == "__main__":
    main()
