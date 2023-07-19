from bs4 import BeautifulSoup
import os
import requests
from urllib.parse import urlparse

 


soup = BeautifulSoup("<html><head><title>URL Response Report</title></head><body><ul id='url-list'></ul></body></html>", "html.parser")

 

# URL list to include in the report.html file
urls = [
    'http://www.example.com',
    'http://www.openai.com',
    'http://www.google.com'
]

 

if not os.path.exists('output'):
    os.makedirs('output')

 

# For each URL
for url in urls:
    #Get the domain names from the URL lists
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace('www.', '')

 

    #add that url's to the html report file
    url_list = soup.find('ul', {'id': 'url-list'})
    li_tag = soup.new_tag('li')
    a_tag = soup.new_tag('a')
    a_tag['href'] = f'output/{domain}.txt'
    a_tag.string = domain
    li_tag.append(a_tag)
    url_list.append(li_tag)

 

    #can request to the URL and save given response content to a text file
    response = requests.get(url, verify=False)
    with open(f'output/{domain}.txt', 'w', encoding='utf-8') as file:
        file.write(response.text)

 

# Write the response to new html file
with open('report.html', 'w', encoding='utf-8') as file:
    file.write(str(soup.prettify()))