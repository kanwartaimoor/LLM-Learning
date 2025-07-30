from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer

hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)

text = "I am excited to show Tokenizers in action"
tokens = tokenizer.encode(text)
tokens


len(tokens) # to actually get the number of tokens 

tokenizer.decode(tokens) # when we decode the tokens, we get the original text back along with the special tokens

# we can also decode the tokens by specifying the skip_special_tokens=True
tokenizer.decode(tokens, skip_special_tokens=True)

# we can also decode the tokens by specifying the skip_special_tokens=True
tokenizer.decode(tokens, skip_special_tokens=True)

tokenizer.vocab # to get the vocabulary of the tokenizer all of the tokens that are in the vocabulary


tokenizer.get_added_vocab() # to get the special tokens or reserved tokens that are in the vocabulary

