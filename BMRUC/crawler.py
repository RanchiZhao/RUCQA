import requests
from bs4 import BeautifulSoup
import time

def get_news_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        title = soup.find('div', class_='nc_title').text.strip()
        content = soup.find('div', class_='nc_body').text.strip()
    except:
        return{
        'title': '',
        'content': ''
    }

    return{
        'title': title,
        'content': content
    }

def get_news_links(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    content_col_2_list = soup.find('div', class_='content_col_2_list')
    try:
        links = content_col_2_list.find_all('a', href=True)
    except:
        return []
    news_links = [link['href'] for link in links if link['href'].startswith('https://news.ruc.edu.cn/archives/')]
    return news_links



def main():
    base_url = 'https://news.ruc.edu.cn/archives/date/2023/'
    for month in range(1, 6):
        page = 1
        while True:
            url = base_url + str(month) + '/page/' + str(page)
            news_links = get_news_links(url)
            if not news_links:  # if there are no news links on this page, stop crawling this month
                break
            print(news_links)
            for link in news_links:
                news_content = get_news_content(link)
                if not news_content['title'] or not news_content['content']:
                    continue
                print(news_content)
                # save the news content to a txt file
                news_id = link.split('/')[-1]  # get the news id from the link
                with open(f'./docs/{news_id}.txt', 'w', encoding='utf-8') as f:
                    f.write(news_content['title'] + '\n' + news_content['content'])
            page += 1

if __name__ == '__main__':
    main()
