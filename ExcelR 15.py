#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install requests beautifulsoup4


# In[8]:


import requests
from bs4 import BeautifulSoup
url="https://quotes.toscrape.com/"
response=requests.get(url)
if response.status_code==200:
    soup=BeautifulSoup(response.text, 'html.parser')
    quotes=soup.find_all("div",class_="quote")
    for i,quote in enumerate(quotes[:5]):
        text=quote.find("span",class_="text").text
        author=quote.find("small",class_="author").text
        tags=[tag.text for tag in quote.find_all("a",class_="tag")]
        print(f"{i+1}.\"{text}\"-{author}")
        print(f" Tags:{','.join(tags)}\n")
else:
    print(f"Failed to retrive the webpage.Status Code: {response.status_code}")


# In[9]:


import requests
from bs4 import BeautifulSoup
city = "india/hyderabad"
url = f"https://www.timeanddate.com/weather/{city}"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
temp = soup.find("div", class_="h2").text.strip() if soup.find("div", class_="h2") else "N/A"
desc = soup.find("p").text.strip() if soup.find("p") else "N/A"
print(f"Current Weather in Hyderabad: {temp} | {desc}")


# In[10]:


import requests
from bs4 import BeautifulSoup
city = "india/hyderabad"
url = f"https://www.timeanddate.com/weather/{city}"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
temp = soup.find("div", class_="h2").text.strip() if soup.find("div", class_="h2") else "N/A"
print(f"Current Weather in Hyderabad: {temp} | {desc}")
desc = soup.find("p").text.strip() if soup.find("p") else "N/A"
print(f"Current Weather in Hyderabad: {temp} | {desc}")


# In[12]:


import requests
from bs4 import BeautifulSoup
search_url = "https://www.amazon.in/s?k=iphone&crid=PQVCJSNI5AH4&sprefix=iphone%2Caps%2C233&ref=nb_sb_noss_2"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(search_url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
product = soup.select_one(".s-title-instructions-style")
price = soup.select_one(".a-price-whole")
if product and price:
    print(f"Product: {product.text.strip()}")
    print(f"Price: Rs.{price.text.strip()}")
else:
    print("Could not find product details.")


# In[19]:


import requests
from bs4 import BeautifulSoup
url="https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population"
response=requests.get(url,headers=headers)
soup=BeautifulSoup(response.text,"html.parser")
table=soup.find("table", class_="wikitable")
for row in table.find_all("tr")[1:6]:
    columns=row.find_all("td")
    country=columns[1].text.strip()
    population=columns[2].text.strip()
    print(f"{country}:{population}")


# In[20]:





# In[ ]:




