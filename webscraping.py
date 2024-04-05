
import requests
from bs4 import BeautifulSoup
 
 
url = 'https://www.trulia.com/CA/San_Jose/'
reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, 'html.parser')
 
print(soup)
urls = []
for link in soup.find_all('a'):
    print(link.get('href'))
    urls.append(link.get('href'))

reqs1=requests.get("https://www.zillowgroup.com/developers/api/public-data/public-records-api/")
