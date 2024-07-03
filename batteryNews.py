#%%
import requests
import pandas as pd
from lxml import html
import time
import re
import json
import trafilatura
from transformers import pipeline
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Function to retry downloading the model
def retry_model_load(model_name, num_retries=3):
    for attempt in range(num_retries):
        try:
            summarizer = pipeline("summarization", model=model_name)
            return summarizer
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < num_retries - 1:
                print("Retrying...")
                time.sleep(5)
            else:
                raise



#%% Function to fetch and parse news links from a given page URL
def fetch_news_links(page_url):
    try:
        response = requests.get(page_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []

    tree = html.fromstring(response.content)

    categories = tree.xpath('//div[@class="list-top-tit"]/p/text()')
    titles = tree.xpath('//div[@class="list-info-con"]/p[@class="font-score"]/text()')
    urls = tree.xpath('//ul[@class="gallery-list clearfix"]/li/a/@href')
    dates = tree.xpath('//span[@class="list-day"]/text()')

    news_links = []
    for category, title, url, date in zip(categories, titles, urls, dates):
        if not url.startswith('http'):
            url = 'https://www.sneresearch.com' + url
        # Extract the source from the end of the title
        source_match = re.findall(r'\[(.*?)\]', title)
        source = source_match[-1] if source_match else ''
        news_links.append((url, category.strip(), title.strip(), date.strip(), source))
    return news_links

# Function to extract details from a single news article
def extract_news_details(url):
    try:
        html_content = trafilatura.fetch_url(url)
        if html_content is None:
            print(f"Failed to fetch the article from {url}")
            return ''
        article_text = trafilatura.extract(html_content, output_format='json')
        if article_text is None:
            print(f"Failed to extract the article text from {url}")
            return ''
        article_json = json.loads(article_text)
        # print(article_json['text'])
        text = article_json.get('text', '')
        return text
    except Exception as e:
        print(f"Error extracting news details: {e}")
        return ''

#%% Function to summarize a text
# Define the summarize_text function
def summarize_text(summarizer, text):
    # Print the input text for debugging
    print("Input text:", text)
    
    # Remove special characters and newlines
    cleaned_text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace characters with a single space
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)  # Replace newlines with a space
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)  # Remove any content within brackets

    # Print cleaned text for debugging
    print("Cleaned text:", cleaned_text)
    
    # Check the length of the text
    if len(cleaned_text.split()) < 50:  # Skip summarization for very short texts
        return cleaned_text

    # Chunk the text if it is too long
    max_chunk_size = 1000  # Adjust the chunk size as needed
    if len(cleaned_text) > max_chunk_size:
        chunks = [cleaned_text[i:i+max_chunk_size] for i in range(0, len(cleaned_text), max_chunk_size)]
    else:
        chunks = [cleaned_text]
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            print("Summary result:", summary)  # Print the summary result for debugging
            
            # Check if the summary list is not empty and has the expected structure
            if summary and isinstance(summary, list) and len(summary) > 0 and 'summary_text' in summary[0]:
                summaries.append(summary[0]['summary_text'])
            else:
                raise ValueError("Unexpected summary structure")
        except Exception as e:
            print(f"Error in summarizing text: {e}")
            summaries.append(chunk)
    
    # Join the summaries of all chunks
    final_summary = ' '.join(summaries)
    return final_summary

#%% Prepare a list to store detailed news data
if __name__ == '__main__':
    news_data = []
    #%% Base URL for the news list
    base_url = 'https://www.sneresearch.com/kr/insight/news/'
    # Initialize the summarizer with retries
    summarizer = retry_model_load("facebook/bart-large-cnn")
    # Retry strategy for requests
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        # method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    # Loop through the first 10 pages to fetch news links and details
    for page in range(0, 120, 12):  # Pages increment by 12 (0, 12, 24, ..., 108)
        page_url = f'{base_url}{page}?s_cat=%7C&s_keyword=#ac_id'
        try:
            news_links = fetch_news_links(page_url)
        except Exception as e:
            print(f"Failed to fetch news links from {page_url}: {e}")
            continue
 
        for url, category, title, date, source in news_links:
            # try:
            # 뉴스 기사중 text만 가져옴
            details = extract_news_details(url)
            if details and title:
                # summary = summarize_text(summarizer, details)
                news_data.append({
                    'URL': url,
                    'Category': category,
                    'Title': title,
                    'Date': date,
                    'Source': source,
                    'Details': details,
                    # 'Summary': summary
                })
                print(title)
            else:
                print(f"No details or title for {url}")
            # except Exception as e:
            #     print(f"Failed to fetch details for {url}: {e}")
            
        # Adding a delay to avoid overwhelming the server
        time.sleep(2)

    #%% Create a DataFrame from the news data
    df = pd.DataFrame(news_data)

    #%% Save DataFrame to an Excel file
    excel_filename = 'detailed_news_with_summary.xlsx'
    df.to_excel(excel_filename, index=False)

    print(f"Data has been written to {excel_filename}")
    