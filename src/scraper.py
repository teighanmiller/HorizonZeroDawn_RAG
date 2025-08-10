"""
Scrapes Horizon Zero Dawn Fan Wiki pages for data.
"""

import csv
import time
from typing import Tuple
from datetime import datetime
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup, PageElement, Tag, NavigableString

BASE_URL = "https://horizon.fandom.com"
TARGET_URL = f"{BASE_URL}/wiki/Special:AllPages"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_content(soup: BeautifulSoup) -> str:
    """
    Gets the content of the webpage

    Args:
        soup (BeautifulSoup): the soup object that contains the webpage html.

    Returns:
        content: The content of the webpage.
    """
    infobox = "[ Infobox source ]"
    content_parts = []
    content_div = soup.find("div", class_="mw-parser-output")
    if content_div:
        for p in content_div.find_all(["p", "ul", "ol"], recursive=False):
            text = p.get_text(" ", strip=True)
            if text:
                content_parts.append(text)
    content = "\n".join(content_parts)

    # Finds the index where "[ Infobox source ]" starts
    idx = content.find(infobox)

    # Returns the string without the content before "[ Infobox source ]"
    if idx != -1:
        return content[idx + len(infobox) :].strip()

    return content.strip()


def get_location(soup: BeautifulSoup) -> str:
    """
    Gets the in game location of the item/character/place described in the webpage.

    Args:
        soup (BeautifulSoup): Object containing webpage html and content

    Returns:
        str: the location of the item/character/place
    """
    loc_div = soup.find(attrs={"data-source": "location"})
    if loc_div:
        a = loc_div.find("a")
        return a.get_text(strip=True) if a else loc_div.get_text(strip=True)
    return ""


def get_category(soup: BeautifulSoup) -> str:
    """
    Gets the labeled category of a page.

    Args:
        soup (BeautifulSoup): Object containing webpage html and content

    Returns:
        str: the category of the page
    """
    cat_div = soup.find(attrs={"data-source": "category"})
    if cat_div:
        val = cat_div.find(class_="pi-data-value") or cat_div
        a = val.find("a")
        return a.get_text(strip=True) if a else val.get_text(" ", strip=True)
    foot = soup.select_one("#mw-normal-catlinks ul li a")
    return foot.get_text(strip=True) if foot else ""


def get_infobox_data(soup: BeautifulSoup) -> dict:
    """
    Gets the content of the info box if it exists.

    Args:
        soup (BeautifulSoup): object containing webpage content

    Returns:
        dict: object containing the contents of the infobox
    """
    data = {}
    for field in soup.select(".pi-data"):
        key_el = field.find(class_="pi-data-label")
        val_el = field.find(class_="pi-data-value")
        if key_el and val_el:
            data[key_el.get_text(strip=True)] = val_el.get_text(" ", strip=True)
    return data


def get_html(
    url: str,
) -> Tuple[BeautifulSoup, PageElement | Tag | NavigableString | None]:
    """
    Gets the content of a webpage.

    Args:
        url (str): the url pointing to the webpage to be scraped.

    Returns:
        BeautifulSoup, PageElement | Tag | Navigable String | None:
        Gets the BeautifulSoup object that contains the webpage and the link to the next page.
    """
    page = requests.get(url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(page.text, "html.parser")
    follow = soup.find("a", string=lambda text: text and text.startswith("Next page"))
    return soup, follow


def get_pages(writer: csv.writer, soup: BeautifulSoup):
    """
    Extracts data from webpage and writes it to a csv

    Args:
        writer (csv.writer): the writer object that allows data to be written to the csv
        soup (BeautifulSoup): the BeautifulSoup object containing the webpage data
    """
    page_links = soup.select(".mw-allpages-body a")
    for link_tag in tqdm(page_links, desc="Processing pages"):
        relative_link = link_tag.get("href")
        page_url = f"{BASE_URL}{relative_link}"
        page = requests.get(page_url, headers=HEADERS, timeout=30)
        new_soup = BeautifulSoup(page.text, "html.parser")

        writer.writerow(
            {
                "url": page_url,
                "category": get_category(new_soup),
                "location": get_location(new_soup),
                "content": get_content(new_soup),
                # **get_infobox_data(new_soup),
            }
        )
        time.sleep(0.5)


def scrape_data():
    """
    Main function for scraping data and writing it to a csv
    """
    # Get the current datetime object
    current_datetime = datetime.now()
    date_time_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    csv_path = f"/Users/teighanmiller/development/courses/zoomcamp/HorizonZeroDawn_RAG/data/horizon_data_{date_time_string}.csv"

    # Format the datetime object into a string
    main_soup, next_page = get_html(TARGET_URL)

    print("Starting ingestion....")
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        field_names = ["url", "category", "location", "content"]
        writer = csv.DictWriter(f=csv_file, fieldnames=field_names)
        writer.writeheader()

        while True:
            get_pages(writer=writer, soup=main_soup)
            if not next_page:
                break
            next_url = f"{BASE_URL}/{next_page.get('href')}"
            main_soup, next_page = get_html(next_url)

    print("Finished data ingestion.")


if __name__ == "__main__":
    scrape_data()
