# Looking behind the curtain of GPT!
# You can simply scroll down to the diagram below to see the results.

# If you'd like to try this yourself - this runs nicely on a free Colab CPU box.

# set the Open AI API key as a 'Secret' in this colab, which will be private for you
# Press the key symbol in the left sidebar
# Enter a Name of OPENAI_API_KEY and paste your actual key in
# Ensure the switch for "Notebook access" is switched on.
# Then you're ready for showtime..


# first install the libraries for OpenAI and for the visualizer

# pip install -q openai networkx

# Get the api key from the secrets

from dotenv import load_dotenv
import os 

# from google.colab import userdata

# api_key = userdata.get('OPENAI_API_KEY')

#ENVIORNMENT
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')


if api_key.startswith("sk-proj-"):
  print("API key looks good so far")
else:
  print("Potential problem finding API key - please check setup instructions!")

  # Check OpenAI connectivity with a not-so taxing question

from openai import OpenAI
openai = OpenAI(api_key=api_key)
response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "What is 2+2?"}])
print(response.choices[0].message.content)


# Some imports
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
import math


# A class to call OpenAI and return a list of dicts of predictions to display

class TokenPredictor:
  def __init__(self, client, model_name: str, temperature: int):
    self.client = client
    self.messages = []
    self.predictions = []
    self.model_name = model_name
    self.temperature = temperature

  def predict_tokens(self, prompt: str, max_tokens: int = 100) -> List[Dict]:
    """
    Generate text token by token and track prediction probabilities.
    Returns list of predictions with top token and alternatives.
    """
    response = self.client.chat.completions.create(
      model=self.model_name,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=max_tokens,
      temperature=self.temperature,
      logprobs=True,
      seed=42,
      top_logprobs=7,
      stream=True
    )

    predictions = []
    for chunk in response:
      if chunk.choices[0].delta.content:
        token = chunk.choices[0].delta.content
        logprobs = chunk.choices[0].logprobs.content[0].top_logprobs
        logprob_dict = {item.token: item.logprob for item in logprobs}

        # Get top predicted token and probability
        top_token = token
        top_prob = logprob_dict[token]

        # Get alternative predictions
        alternatives = []
        for alt_token, alt_prob in logprob_dict.items():
          if alt_token != token:
            alternatives.append((alt_token, math.exp(alt_prob)))
        alternatives.sort(key=lambda x: x[1], reverse=True)

        prediction = {'token': top_token, 'probability': math.exp(top_prob),'alternatives': alternatives[:2]}
        predictions.append(prediction)

    return predictions



# A function to create a directed graph based on the predictions

def create_token_graph(model_name:str, predictions: List[Dict]) -> nx.DiGraph:
  """
  Create a directed graph showing token predictions and alternatives.
  """
  G = nx.DiGraph()

  G.add_node("START", token=model_name, prob="START", color='lightgreen', size=4000)

  # First, create all main token nodes in sequence
  for i, pred in enumerate(predictions):
    token_id = f"t{i}"
    G.add_node(token_id, token=pred['token'], prob=f"{pred['probability']*100:.1f}%", color='lightblue', size=6000)
    G.add_edge(f"t{i-1}" if i else "START", token_id)

  # Then add alternative nodes with a different y-position
  last_id = None
  for i, pred in enumerate(predictions):
    parent_token = "START" if i == 0 else f"t{i-1}"

    # Add alternative token nodes slightly below main sequence
    for j, (alt_token, alt_prob) in enumerate(pred['alternatives']):
      alt_id = f"t{i}_alt{j}"
      G.add_node(alt_id, token=alt_token, prob=f"{alt_prob*100:.1f}%", color='lightgray', size=6000)
      G.add_edge(parent_token, alt_id)

  G.add_node("END", token="END", prob="100%", color='red', size=6000)
  G.add_edge(parent_token, "END")
  return G



    # Visualize the graph using Matplotlib

def visualize_predictions(G: nx.DiGraph, figsize=None):
  """
  Visualize the token prediction graph with vertical layout and better spacing.
  """
  num_nodes = len(G.nodes)
  spacing_y = 15  # Increased spacing for vertical layout
  spacing_x = 8   # Increased horizontal spacing for alternatives

  # Dynamically set figure size based on number of nodes
  if figsize is None:
    figsize = (10, max(10, num_nodes // 2))

  plt.figure(figsize=figsize)

  pos = {}
  main_nodes = [n for n in G.nodes if "_alt" not in n]

  # Position main token nodes in a vertical line
  for i, node in enumerate(main_nodes):
    pos[node] = (0, -i * spacing_y)  

  # Position alternative nodes at varying offsets
  for node in G.nodes:
    if "_alt" in node:
      main_token = node.split("_")[0]
      alt_num = int(node.split("_alt")[1])
      if main_token in pos:
        x_offset = (-1 if alt_num % 2 == 0 else 1) * (spacing_x + alt_num * 2)
        pos[node] = (x_offset, pos[main_token][1] + 3)  # Slightly above main token

  # Draw nodes with their assigned positions
  node_colors = [G.nodes[node]["color"] for node in G.nodes]
  node_sizes = [G.nodes[node]["size"] for node in G.nodes]
  nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

  # Draw edges as straight lines
  nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, alpha=0.7)

  # Add labels with tokens and probabilities
  labels = {node: f"{G.nodes[node]['token']}\n{G.nodes[node]['prob']}" for node in G.nodes}
  nx.draw_networkx_labels(G, pos, labels, font_size=10)

  plt.title("how would you describe color blue")
  plt.axis("off")

  # Auto-adjust plot limits
  margin = 10
  x_values = [x for x, y in pos.values()]
  y_values = [y for x, y in pos.values()]
  plt.xlim(min(x_values) - margin, max(x_values) + margin)
  plt.ylim(min(y_values) - margin, max(y_values) + margin)


      # Save the plot as a PNG file
  plt.savefig("abc", dpi=300, bbox_inches="tight")
  plt.close()  



# And now, pick a model and run!

# For advanced experimenters: try changing temperature to a higher number like 0.4
# A higher temperature means that occasionally a token other than the one with the highest probability will be picked
# Resulting in a more diverse output

model_name = "gpt-4o"
temperature = 0.0

predictor = TokenPredictor(openai, model_name, temperature)
prompt = "How would you describe the color blue to someone who has never been able to see, in no more than 3 sentences"
predictions = predictor.predict_tokens(prompt)
G = create_token_graph(model_name, predictions)
visualize_predictions(G)
