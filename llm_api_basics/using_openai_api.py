import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI
import gradio as gr


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


system_prompt = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

user_prompt = "Hey Kanwar! how are you doing today? did you start learning AI?"





def call_gpt(prompt):

    message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]


    stream = openai.chat.completions.create(
            model = "gpt-4o-mini",
            messages = message,
            stream = True
        )
        
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ''
        yield result


view = gr.Interface(

  fn=call_gpt,
  inputs=[
    gr.Textbox(label="say something:")
  ],
  outputs=[gr.Markdown(label="output")],
  flagging_mode="never"
)

view.launch(share=True)



  