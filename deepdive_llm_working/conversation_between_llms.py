#we will have a conversation between llama and gpt 

# imports
import requests
from openai import OpenAI
import ollama 
import os
from dotenv import load_dotenv

#ENV
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

openai = OpenAI()

# Constants
LAMA_MODEL = "llama3.2"
gpt_model = "gpt-4o-mini"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

lama_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

gpt_messages = ["Hi there"]
lama_messages = ["Hi"]


#Functionality
def call_gpt():
  messages = [{"role": "system", "content": gpt_system}]


  for gpt, lama in zip(gpt_messages, lama_messages):
    messages.append({"role": "assistant", "content": gpt})
    messages.append({"role": "user", "content": lama})
  
  print("\n message going in GPT")
  print(messages)
  print("\n GPT OUT")

  completion = openai.chat.completions.create(
      model=gpt_model,
      messages=messages
  )
  return completion.choices[0].message.content


def call_lama():
  messages = []

  for gpt, lama_message in zip(gpt_messages, lama_messages):
    messages.append({"role": "user", "content": gpt})
    messages.append({"role": "assistant", "content": lama_message})
  
  messages.append({"role": "user", "content": gpt_messages[-1]})
  
  print("\n message going in LAMA")
  print(messages)
  print("\n LAMA OUT")


  response = ollama.chat(model=LAMA_MODEL, messages=messages)
  return response['message']['content']


gpt_messages = ["Hi there"]
claude_messages = ["Hi"]

print(f"GPT:\n{gpt_messages[0]}\n")
print(f"LAMA:\n{claude_messages[0]}\n")

for i in range(5):
  gpt_next = call_gpt()
  print(f"GPT:\n{gpt_next}\n")
  
  gpt_messages.append(gpt_next)
  
  lama_next = call_lama()
  print(f"LAMA:\n{lama_next}\n")
  lama_messages.append(lama_next)






