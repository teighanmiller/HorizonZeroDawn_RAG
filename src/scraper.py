"""
Scrapes Horizon Zero Dawn Fan Wiki pages for data.
"""

import csv
import time
from random import random
from typing import Tuple
from datetime import datetime
from urllib.parse import urljoin
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup, PageElement, Tag, NavigableString
from ollama import chat, ChatResponse

BASE_URL = "https://horizon.fandom.com"
TARGET_URL = f"{BASE_URL}/wiki/Special:AllPages"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/118.0.0.0 Safari/537.36",
    "Referer": "https://horizon.fandom.com/wiki/Special:AllPages",
}


def classify_page(title: str, text: str) -> str:
    """
    Classifies the content of the webpage into machine, society, location, object, character or other categories.

    Args:
        title (str): the url of the webpage.
        text (str): the content of the webpage.

    Returns:
        str: the classification of the webpage.
    """
    response: ChatResponse = chat(
        model="gemma3",
        messages=[
            {
                "role": "user",
                "content": f"""
                    Classify the webpage and its content into one of the following categories:
                    - machine: this category contains information about machines in Horizon.
                    - society: this category contains information about the cultures and peoples in Horizon.
                    - location: this category contains information about specific locations and cities in the game.
                    - object: this category contains information about in game objects.
                    - character: this category contains information about specific characters.
                    - other: this category contains information that does not fit into the other categories.

                    This is the webpage url: {title}
                    This is the webpage content:
                    {text}

                    return only one of the following classifications: machine, society, location, object, character, or other
                """,
            },
        ],
    )
    return response["message"]["content"].strip()


def get_content(soup: BeautifulSoup) -> list:
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
        return [content[idx + len(infobox) :].strip()]

    return [content.strip()]


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
    Fetch a Special:AllPages page and return its soup and the 'Next page' <a> tag if present.
    Robust to nested markup inside the anchor.
    """
    html = safe_get(url)
    if not html or html == "None":
        return BeautifulSoup("", "html.parser"), None

    soup = BeautifulSoup(html, "html.parser")

    # Prefer anchors that are clearly part of AllPages nav; fall back to any with the title.
    candidates = soup.select(
        '.mw-allpages-nav a[title="Special:AllPages"]'
    ) or soup.select('a[title="Special:AllPages"]')

    follow = None
    for a in candidates:
        if a.get_text(strip=True).startswith("Next page"):
            follow = a
            break

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
        category = get_category(new_soup)
        location = get_location(new_soup)
        content = get_content(new_soup)
        classification = classify_page(page_url, content[0])

        if not content:
            content = [""]

        if len(content[0].split()) > 500:
            content = batch(content[0])

        for item in content:
            new_content = [page_url, classification, category, location, item]
            writer.writerow(new_content)
        time.sleep(1.5 + random())


def batch(content: str) -> list:
    """
    Split long strings by paragraphs.

    Args:
        content (str): a string containing more than 500 words

    Returns:
        list: the string split into paragraphs
    """
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    return paragraphs


def safe_get(url: str, retries: int = 3, backoff: float = 2.0) -> str:
    """
    Makes a GET request with retries and error handling.

    Args:
        url (str): The URL to fetch.
        retries (int): Number of retries before giving up.
        backoff (float): Delay between retries (seconds).

    Returns:
        str | None: Response text if successful, otherwise None.
    """
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()  # Raises HTTPError for bad responses
            return response.text
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(backoff * attempt)  # Exponential backoff
            else:
                return "None"


def scrape_data() -> str:
    """
    Main function for scraping data and writing it to a csv

    Returns:
    str: path to the output data
    """
    # Get the current datetime object
    current_datetime = datetime.now()
    date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    csv_path = f"data/horizon_data_{date_time_string}.csv"

    # Format the datetime object into a string
    main_soup, next_page = get_html(TARGET_URL)

    print("Starting ingestion....")
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        field_names = ["url", "classification", "category", "location", "content"]
        writer = csv.writer(csv_file)
        writer.writerow(field_names)

        while True:
            print(next_page)
            get_pages(writer=writer, soup=main_soup)
            if not next_page:
                break
            next_url = urljoin(BASE_URL, next_page.get("href", ""))
            main_soup, next_page = get_html(next_url)
    print("Finished data ingestion.")
    return csv_path


if __name__ == "__main__":
    scrape_data()
