import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI


# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Check the key
if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("sk-proj-"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")


openai = OpenAI()


system_prompt = "You are an assistant that analyzes the contents of a text \
and provides a response to that text"

user_prompt = "Hey Kanwar! how are you doing today? did you start learning AI?"


message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = message
    )
    
print (response.choices[0].message.content)