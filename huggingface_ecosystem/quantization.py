from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc


# instruct models
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-2b-it"
QWEN2 = "Qwen/Qwen2-7B-Instruct" 
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1" 

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]


#Hugging Face Login
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)


# Quantization Config - this allows us to load the model into memory and use less memory

# paramter is in 32 bit 4 bytes it loads it in just 4 bit s

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLAMA)

tokenizer.pad_token = tokenizer.eos_token
# pad token is which token is used to pad the sequence to the same length or fill up the prompt 
#its a common practise to set pad token to eos token/ end of sequence token
#matrix operations are possible with pad token
# Suppose you have 2 sentences:

# "I love AI" → 3 tokens

# "ChatGPT is amazing" → 4 tokens

# To process them together as a batch, they must be the same length. So you pad the shorter one:

# arduino
# Copy
# Edit
# "I love AI [PAD]"  → 4 tokens  
# "ChatGPT is amazing" → 4 tokens


inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")


# The model

model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
# automodelforcausallm is a model that is used for causal language modeling. its a basic class for llms 
# we made a tokeniser and then we made a model from it 
#device map is used to map the model to the device for eg if we have a gpu we can map the model to the gpu to use that 



outputs = model.generate(inputs, max_new_tokens=80)
print(tokenizer.decode(outputs[0]))


