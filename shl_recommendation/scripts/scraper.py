"""
SHL Product Catalogue Scraper
Scrapes all Individual Test Solutions from https://www.shl.com/solutions/products/product-catalog/
Run this first before anything else: python scripts/scraper.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import os

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_PATH = "data/shl_assessments.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_soup(url: str, retries: int = 3) -> BeautifulSoup:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            print(f"  [Attempt {attempt+1}] Error fetching {url}: {e}")
            time.sleep(2)
    return None


def scrape_catalogue_page(page_num: int = 0, per_page: int = 12) -> dict:
    """Scrape one page of the catalogue (Individual Test Solutions only)."""
    params = {
        "start": page_num * per_page,
        "type": 1,  # 1 = Individual Test Solutions, 2 = Pre-packaged Job Solutions
    }
    url = CATALOG_URL
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    soup = BeautifulSoup(resp.text, "lxml")
    return soup


def parse_assessment_cards(soup: BeautifulSoup) -> list:
    """Extract assessment basic info from catalogue listing page."""
    assessments = []
    # SHL catalogue uses table rows or card divs — try both patterns
    rows = soup.select("tr.catalogue__row") or soup.select("[data-course-id]")
    
    if not rows:
        # fallback: find all links that point to product-catalog/view/
        links = soup.find_all("a", href=lambda h: h and "/product-catalog/view/" in h)
        for link in links:
            href = link.get("href", "")
            full_url = BASE_URL + href if href.startswith("/") else href
            name = link.get_text(strip=True)
            if name:
                assessments.append({"name": name, "url": full_url})
        return assessments

    for row in rows:
        a_tag = row.find("a", href=True)
        if not a_tag:
            continue
        href = a_tag["href"]
        full_url = BASE_URL + href if href.startswith("/") else href
        name = a_tag.get_text(strip=True)
        assessments.append({"name": name, "url": full_url})

    return assessments


def scrape_assessment_detail(url: str) -> dict:
    """Visit each assessment page and scrape detailed info."""
    soup = get_soup(url)
    if not soup:
        return {}

    detail = {"url": url}

    # Name
    h1 = soup.find("h1")
    detail["name"] = h1.get_text(strip=True) if h1 else ""

    # Description — look for the main content paragraph
    desc_candidates = soup.select(".product-catalogue-training-calendar__row--description p") or \
                      soup.select(".catalogue__detail-description p") or \
                      soup.select("article p") or \
                      soup.select("main p")
    if desc_candidates:
        detail["description"] = " ".join(p.get_text(strip=True) for p in desc_candidates[:3])
    else:
        detail["description"] = ""

    # Test type badges (A, B, C, P, K, S, E, D)
    test_type_tags = soup.select(".product-catalogue__key") or \
                     soup.select("[class*='test-type']") or \
                     soup.select(".badge")
    type_texts = [t.get_text(strip=True) for t in test_type_tags if t.get_text(strip=True)]

    # Map single letter codes to full names
    type_map = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations",
    }
    full_types = []
    for t in type_texts:
        full_types.append(type_map.get(t, t))
    detail["test_type"] = full_types if full_types else []

    # Duration
    duration = None
    for tag in soup.find_all(string=True):
        text = tag.strip()
        if "minute" in text.lower() and any(c.isdigit() for c in text):
            import re
            nums = re.findall(r"\d+", text)
            if nums:
                duration = int(nums[0])
                break
    detail["duration"] = duration

    # Remote / Adaptive support
    page_text = soup.get_text().lower()
    detail["remote_support"] = "Yes" if "remote" in page_text else "No"
    detail["adaptive_support"] = "Yes" if "adaptive" in page_text else "No"

    # Job levels
    job_levels = []
    for kw in ["graduate", "entry", "mid-level", "manager", "director", "executive", "professional"]:
        if kw in page_text:
            job_levels.append(kw.title())
    detail["job_levels"] = job_levels

    return detail


def scrape_all_catalogue() -> pd.DataFrame:
    """
    Main scrape loop — goes through all paginated pages of
    Individual Test Solutions (type=1) and collects every assessment.
    """
    all_assessments = []
    page = 0
    per_page = 12

    print("=" * 60)
    print("SHL CATALOGUE SCRAPER")
    print("=" * 60)

    while True:
        start = page * per_page
        url = f"{CATALOG_URL}?start={start}&type=1"
        print(f"\n[Page {page+1}] Fetching: {url}")

        soup = get_soup(url)
        if not soup:
            print("  Failed to get page, stopping.")
            break

        cards = parse_assessment_cards(soup)
        if not cards:
            print("  No more assessments found. Done paginating.")
            break

        print(f"  Found {len(cards)} assessments on this page.")
        all_assessments.extend(cards)

        # Check if there's a "next" page
        next_btn = soup.find("a", string=lambda s: s and "next" in s.lower()) or \
                   soup.select_one("[aria-label='Next']") or \
                   soup.select_one(".pagination__next:not(.disabled)")
        if not next_btn:
            print("  No next page found. Done.")
            break

        page += 1
        time.sleep(1)  # be polite

    print(f"\nTotal assessments found: {len(all_assessments)}")
    print("\nNow scraping detail pages...")

    detailed = []
    for i, item in enumerate(all_assessments):
        print(f"  [{i+1}/{len(all_assessments)}] {item.get('name', item['url'])}")
        detail = scrape_assessment_detail(item["url"])
        # Merge basic + detail
        merged = {**item, **detail}
        detailed.append(merged)
        time.sleep(0.5)

    df = pd.DataFrame(detailed)

    # Clean up
    df["test_type"] = df["test_type"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    df["job_levels"] = df["job_levels"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    df = df.drop_duplicates(subset=["url"])
    df = df[df["name"].str.len() > 0]

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(df)} assessments to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    df = scrape_all_catalogue()
    print(df[["name", "url", "test_type", "duration"]].head(10).to_string())
