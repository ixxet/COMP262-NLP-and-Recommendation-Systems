"""
COMP 262 - Assignment 1, Exercise 1: Web Scraping
Student: Izzet Abidi (300898230)

Scrapes the Centennial College AI program page to extract:
- Website title
- Program highlights
- Companies offering jobs
- Career outlook
Exports the results to Izzet_my_future.csv

Uses Selenium for dynamic page rendering since the site is built with React.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import csv
import os
import time

# URL of the Centennial College AI program page
URL = "https://www.centennialcollege.ca/programs-courses/full-time/artificial-intelligence"


def fetch_page(url):
    """Launches a headless browser to render the dynamic page and returns the parsed HTML."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    # Wait for the page content to load
    time.sleep(5)

    # Click the "Career Options and Education Pathways" tab to reveal full content
    try:
        career_tab = driver.find_element(By.XPATH, '//*[contains(text(), "Career Options")]')
        career_tab.click()
        time.sleep(3)
    except Exception:
        print("Note: Could not click Career Options tab, proceeding with available content.")

    soup = BeautifulSoup(driver.page_source, "html.parser")
    title = driver.title
    driver.quit()
    return soup, title


def get_program_highlights(page_text):
    """Extracts program highlights from the full page text."""
    highlights = []
    lines = page_text.split("\n")

    capture = False
    for line in lines:
        line = line.strip()
        if line == "Program Highlights":
            capture = True
            continue
        if capture:
            # Stop when reaching the next section heading
            if line in ["Career Outlook", "Education Pathways", "Companies Offering Jobs",
                        "Future Alumni", "Academic Pathways"]:
                break
            if line and len(line) > 10:
                highlights.append(line)

    return highlights


def get_companies(page_text):
    """Extracts companies offering jobs from the page text."""
    companies = []
    lines = page_text.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("IBM Canada") or (line.startswith("Companies Offering Jobs") is False
                                              and "IBM" in line and "Bell" in line):
            # The companies are listed as a comma-separated string
            companies = [c.strip() for c in line.replace(" and more.", "").split(",") if c.strip()]
            break

    # Fallback: search after "Companies Offering Jobs" heading
    if not companies:
        capture = False
        for line in lines:
            line = line.strip()
            if "Companies Offering Jobs" in line:
                capture = True
                continue
            if capture and line:
                companies = [c.strip() for c in line.replace(" and more.", "").split(",") if c.strip()]
                break

    return companies


def get_career_outlook(page_text):
    """Extracts career outlook entries from the page text."""
    careers = []
    lines = page_text.split("\n")

    capture = False
    for line in lines:
        line = line.strip()
        if line == "Career Outlook":
            capture = True
            continue
        if capture:
            # Stop at next section
            if line in ["Education Pathways", "Academic Pathways", "Personality Test",
                        "Career Explorer"]:
                break
            if line and len(line) > 3:
                careers.append(line)

    return careers


def export_to_csv(title, highlights, companies, careers, filename):
    """Exports all scraped data to a CSV file."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Information"])

        # Title
        writer.writerow(["Title", title])

        # Program highlights
        for item in highlights:
            writer.writerow(["Program Highlight", item])

        # Companies
        for item in companies:
            writer.writerow(["Company Offering Jobs", item])

        # Career outlook
        for item in careers:
            writer.writerow(["Career Outlook", item])

    print(f"Data exported to {filename}")


def main():
    print("=" * 60)
    print("Exercise 1: Web Scraping - Centennial College AI Program")
    print("=" * 60)

    # Fetch the page using Selenium (dynamic content)
    print(f"\nFetching page: {URL}")
    print("(Rendering with headless browser...)")
    soup, title = fetch_page(URL)

    # Get the full page text for parsing
    body = soup.find("body")
    page_text = body.get_text(separator="\n", strip=True) if body else ""

    # Extract data
    print(f"\n--- Website Title ---\n{title}")

    highlights = get_program_highlights(page_text)
    print(f"\n--- Program Highlights ({len(highlights)} found) ---")
    for i, h in enumerate(highlights, 1):
        print(f"  {i}. {h}")

    companies = get_companies(page_text)
    print(f"\n--- Companies Offering Jobs ({len(companies)} found) ---")
    for i, c in enumerate(companies, 1):
        print(f"  {i}. {c}")

    careers = get_career_outlook(page_text)
    print(f"\n--- Career Outlook ({len(careers)} found) ---")
    for i, c in enumerate(careers, 1):
        print(f"  {i}. {c}")

    # Export to CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "Izzet_my_future.csv")
    export_to_csv(title, highlights, companies, careers, output_file)

    print(f"\nTotal items scraped: {1 + len(highlights) + len(companies) + len(careers)}")
    print("Done.")


if __name__ == "__main__":
    main()
