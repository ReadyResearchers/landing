import requests
from bs4 import BeautifulSoup
import re

def load_job():
    with open('data/jobtitle.txt') as f:
        job_title = f.read().replace('\n', '')
        return job_title


def get_url(position):

    url = f"https://indeed.com/jobs?q={position}"

    return url


def get_record(card):
   
    a_tag = card.h2.a

    job_id = a_tag.get("id")  # job id
    job_title = a_tag.get("aria-label")  # job title
    job_url = 'https://www.indeed.com' + a_tag.get('href')  # job url
    company_name =   # company name
    job_loc =   # job location
    job_summary =  # job description
    job_salary =  # job salaries

    record = (job_id, job_title, job_loc, job_summary, job_salary, job_url, company_name)

    return record


def get_jobs(position):
 
    url = get_url(position)
    records = []

    # extract the job data
    response = ""
    while response == "":
            response = requests.get(url=url)
            break

    soup = BeautifulSoup(response.text, 'html.parser')

    cards = soup.find_all('div', {'class': 'job_seen_beacon'})

    for card in cards:
        record = get_record(card)
        records.append(record)
    
    return record

    
def main():
    job_list = load_job()
    for job in job_list:
        job = job.replace(' ', '%20')
        record = get_jobs(job)
    print(record)