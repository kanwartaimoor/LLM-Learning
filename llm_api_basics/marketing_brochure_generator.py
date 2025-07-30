# Create a product that can generate marketing brochures about a company 
# - for prospective clients 
# - for investors 
# - for rescuitment 

# THE TECH 
# - open ai api 
# - one shot prompting 
# - stream back formatted results


#IMPORTS
import os 
import requests 
import json 
from typing import List 
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI


#CONSTANTS
MODEL='gpt-4o-mini'
headers = {
   "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}


#ENVIORNMENT
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI()

if api_key and api_key.startswith('sk-proj-') and len(api_key) > 10:
  print('API key looks good\n')
else: 
  print('Something wrong with api key!\n')




class Website:
  """
   utility class of website object that we will parse with links 
  """

  def __init__(self, url):
    print("Generating website object... \n \n")

    self.url = url
    response = requests.get(url, headers=headers)
    self.body = response.content

    soup = BeautifulSoup(self.body, 'html.parser')
    self.title = soup.title.string if soup.title else "No title found"

    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)
    else:
        self.text = ""

    links = [link.get('href') for link in soup.find_all('a')]
    self.links = [link for link in links if link]
    print("Generated website object!! \n \n")
  
  def get_contents(self):
    return f"Webpage title:\n {self.title} \n webpage contents:\n {self.text}\n \n"




link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""


def get_links_user_prompt(website):
  print("Generating user prompt to get additional links... \n \n")


  user_prompt = f"Here is the list of links on the website of {website.url} - "
  user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
  user_prompt += "Links (some might be relative links):\n"
  user_prompt += "\n".join(website.links)

  print("Generated user prompt to get additional links!! \n \n")
  return user_prompt


def get_links(url):
  print("Fetching user prompt to get additional links... \n \n")
  website = Website(url)
  response = openai.chat.completions.create(
    model=MODEL,
    messages=[
      {"role": "system", "content": link_system_prompt},
      {"role": "user", "content": get_links_user_prompt(website)}
    ],
    response_format={"type": "json_object"}
  )

  result = response.choices[0].message.content
  print("Fetched user prompt to get additional links!! \n \n")

  return json.loads(result)


def get_all_details(url):
  print("Fetching all the details using all the links... \n \n")

  result = "Landing page: \n"
  result += Website(url).get_contents()
  links = get_links(url)

  for link in links["links"]:
    result += f"\n \n {link['type']} \n"
    result += Website(link["url"]).get_contents()

  print("Fetched all the details using all the links!! \n \n")
  return result


system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

# Or uncomment the lines below for a more humorous brochure - this demonstrates how easy it is to incorporate 'tone':

# system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
# and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
# Include details of company culture, customers and careers/jobs if you have the information."

def get_brochure_user_prompt(company_name, url):
  print("Generating user prompt to create a brochure... \n \n")

  user_prompt = f"You are looking at a company called: {company_name}\n"
  user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company.\n"
  user_prompt += get_all_details(url)
  user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters


  print("Generated user prompt to create a brochure!! \n \n")
  return user_prompt


def create_brochure(company_name, url):

  print(f"Creating the brochure {company_name} ... \n \n")

  response = openai.chat.completions.create(
      model=MODEL,
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
  )
  result = response.choices[0].message.content
  print(f"Created the brochure {company_name} !! \n \n")
  print(result)
  return result

# create_brochure("HuggingFace", "https://huggingface.co")

def stream_brochure(company_name, url):
  print(f"Creating the brochure {company_name} ... \n \n")

  stream = openai.chat.completions.create(
    model=MODEL,
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
    ],
    stream = True
  )
  
  for chunk in stream:
    print( chunk.choices[0].delta.content or '', end='')
  
  print(f" \n Created the brochure {company_name}!! \n \n")

stream_brochure("HuggingFace", "https://huggingface.co")