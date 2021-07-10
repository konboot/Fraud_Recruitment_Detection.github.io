import pandas as pd
import requests
import csv
from bs4 import BeautifulSoup

def extract(page):
    headers = {'Use-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
for i in range(1,10):
    url = "https://companies.naukri.com/course5i/jobs/?othersrcp=38046&wExp=N"
    page = requests.get(url)

    if(page.status_code==200):
        print('Data fetched successfully', i)
        

        soup = BeautifulSoup(page.content, 'html.parser')
        #saving = soup.findAll(attrs = {'class': 'content'}
        results = soup.find(id = "after_section_1")
        job_elements = results.find_all("div", class_="slide-entry-wrap")

        #for job_element in job_elements:
        #   print(job_element, end="\n"*2)

        #title = soup.findAll(attrs={'class': 'slide-entry-wrap'})
        #print (title[0].text.replace('\n', " "))
        #print (title.prettify())

        #for job_element in job_elements:
       #     title_element = job_element.find("h3", class_="slide-entry-title entry-title").text.strip()
      #      expirence_element = job_element.find("span", class_="job-meta slide-meta-exp").text.strip()
     #       salary_element = job_element.find("span", class_="job-meta slide-meta-sal").text.strip()
    #        location_element = job_element.find("span", class_="job-meta slide-meta-loc").text.strip()
            #job_description_element = job_element.find("div", class_="slide-entry-excerpt entry-content ")
            #print(title_element.text.strip())
            #print(expirence_element.text.strip())
            #print(salary_element.text.strip())
            #print(location_element.text.strip())
            #print(job_description_element.text.strip())
            #print()
   #     data = (url, title_element, expirence_element, salary_element, location_element)
    
  #  else:
 #       print('URL not Found', i)

#df = pd.DataFrame(data, columns= ['url', 'title_element', 'expirence_element', 'salary_element', 'location_element'])
#df.to_csv('pagdata.csv')