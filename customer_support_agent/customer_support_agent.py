#establish some ground rules for exmaple if model doesnt know something it should say so instead of hallucinating 
#provide critical background context 

#context building 
#during conversation insert context to give more relevant background information for the topic 

#multishot prompting to train on conversational style and scenarios by providing relevant conversations 
#this is kind of training but also not exactly training by just giving examples its predictions gets strong to get better next token 



# chat(message, history)

# Which expects to receive history in a particular format, which we need to map to the OpenAI format before we call OpenAI:

# [
#     {"role": "system", "content": "system message here"},
#     {"role": "user", "content": "first user prompt here"},
#     {"role": "assistant", "content": "the assistant's response"},
#     {"role": "user", "content": "the new user prompt"},
# ]
# But Gradio has been upgraded! Now it will pass in history in the exact OpenAI format, perfect for us to send straight to OpenAI.

# So our work just got easier!

# We will write a function chat(message, history) where:
# message is the prompt to use
# history is the past conversation, in OpenAI format

# We will combine the system message, history and latest message, then call OpenAI.

#imports 
import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import re

# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")


# Initialize
openai = OpenAI()
MODEL = 'gpt-4o-mini'


system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get."  

system_message += "\nIf the customer asks for shoes, you should respond that shoes are not on sale today, \
but remind the customer to look at hats!"


def chat(message, history):
  relevant_system_message = system_message

  #I understand this a very bad way to check for something but this is just an exmaple of giving additional context 
  #you can say the behaviour is kind of a baby rag application KIND OF 

  # Create a regex pattern to match 'belt' as a whole word, case insensitive
  pattern = r'\bbelt\b'

  if re.search(pattern, message, re.IGNORECASE):
    relevant_system_message += " The store does not sell belts; if you are asked for belts, be sure to point out other items on sale."
  
  messages = [{"role": "system", "content": relevant_system_message}] + history + [{"role": "user", "content": message}]

  
  print("History is:")
  print(history)
  print("And messages is:")
  print(messages)

  stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

  response = ""
  for chunk in stream:
    response += chunk.choices[0].delta.content or ''
    yield response

gr.ChatInterface(fn=chat, type="messages").launch()