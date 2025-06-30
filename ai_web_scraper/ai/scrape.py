import selenium.webdriver as webdriver
from selenium.webdriver.chrome.service import Service

from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time
import os

def scrape_website(website):
    print("Launching chrome browser...")

    load_dotenv()

    # xattr -d com.apple.quarantine chromedriver 

    chrome_driver_path = os.environ.get('CHROME_DRIVER')
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)

    try:
        driver.get(website)
        time.sleep(10)
        print("Page loaded")
        html = driver.page_source

        
        print(html)
        return html
    finally:
        driver.quit()

def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    print(f"Length of body content: {len(body_content)}")
    if body_content:
        return str(body_content)
    return ""

def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")

    for script_or_style in soup(['script', 'style']):
        script_or_style.extract()

    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content

def split_dom_content(dom_content, max_length=6000):
    return [
        dom_content[i : i + max_length] for i in range(0, len(dom_content), max_length)
    ]
