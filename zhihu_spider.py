import requests
from bs4 import BeautifulSoup
import time
import random
import json

base_url = "https://www.yanxuanwk.com"
next_page_url = "https://www.yanxuanwk.com/?ref=codernav.com"

all_links = []

# User-Agent headers list, randomly selected
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
]

# Function to get links from a page
def get_links(url, retries=3):
    headers = {
        'User-Agent': random.choice(user_agents),
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)  # increased timeout
        response.raise_for_status()  # check response status
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get all "typology-button" links
        links = soup.find_all('a', class_='typology-button')
        for link in links:
            full_url = link.get('href')
            if full_url and not full_url.startswith('javascript:'):  # skip JavaScript links
                if not full_url.startswith('http'):  # if not a full URL, prepend base_url
                    full_url = base_url + full_url
                all_links.append(full_url)

        # Find "next" button for pagination
        next_button = soup.find('a', href=True, text="加载更多")
        if next_button:
            next_page_url = next_button['href']
            return next_page_url
        else:
            return None
    except requests.RequestException as e:
        if retries > 0:
            print(f"Request failed: {e}. Retrying...")
            time.sleep(random.uniform(1, 3))
            return get_links(url, retries=retries-1)
        else:
            print(f"Request failed: {e}. No more retries.")
            return None
    except Exception as e:
        print(f"Error: {e} for URL: {url}")
        return None

# Function to fetch content from each link and append to articles list
def fetch_content(url, filename='dataset.json'):
    headers = {
        'User-Agent': random.choice(user_agents),
    }
    retries = 3
    backoff_factor = 2

    try:
        for _ in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)  # Adjust timeout as necessary
                response.raise_for_status()  # Check response status
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e} for URL: {url}")
                time.sleep(backoff_factor ** _)
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('h1', class_='entry-title').get_text(strip=True)
            content_tag = soup.find('div', class_='entry-content')  # Adjust according to the actual HTML structure
            if content_tag:
                text = content_tag.get_text("\n", strip=True)
                article_data = {
                    "name": title,
                    "text": text
                }
                
                # 追加保存到 JSON 文件
                with open(filename, 'a', encoding='utf-8') as f:  # 注意：使用 'a' 模式 (append)
                    json.dump(article_data, f, ensure_ascii=False, indent=2)
                    f.write('\n')  # 添加换行符，以便每个 JSON 对象独占一行
                print(f"Content from {url} appended to {filename}\n")
                return  # Exit function on successful retrieval
            else:
                print(f"No content found at {url}")

        print(f"Failed to fetch content from {url} after {retries} retries")

    except Exception as e:
        print(f"Error: {e} for URL: {url}")


# Loop through pagination until no more pages or retries exceeded
i = 0
while next_page_url:
    print(f"Scraping {next_page_url}")
    next_page_url = get_links(next_page_url)
    if next_page_url:
        time.sleep(random.uniform(1, 3))  # random delay to avoid overwhelming the server
    i += 1
    if i > 10:  # limit to 10 pages for demonstration
        print(f"Found {len(all_links)} links")
        # Scraping content from each link
        for link in all_links:
            fetch_content(link)
            time.sleep(random.uniform(1, 3))  # random delay to avoid overwhelming the server
        i = 0
        all_links = []

print(f"All scraped articles appended to dataset.json") 