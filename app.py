from flask import Flask, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

# Loading real descriptions
def load_real_descriptions():
  with open('real-descriptions.txt', "r", encoding="utf-8") as f:
    return [line.strip() for line in f.readlines()]

real_descriptions = load_real_descriptions()

# Preparing the model to generate fake descriptions
model = GPT2LMHeadModel.from_pretrained('allanjuan/fakemons')

begin_of_sentence_token = '<|startoftext|>'
end_of_sentence_token = '<|endoftext|>'
pad_token = '<|pad|>'

tokenizer = GPT2Tokenizer.from_pretrained(
    'gpt2-medium',
    bos_token=begin_of_sentence_token,
    eos_token=end_of_sentence_token, 
    pad_token=pad_token
)

model.resize_token_embeddings(len(tokenizer))

def generate_fake_descriptions(count):
  generated = tokenizer(f'{begin_of_sentence_token}', return_tensors="pt").input_ids

  sample_outputs = model.generate(
    generated,
    do_sample=True,
    top_k=50, 
    max_length=300,
    top_p=0.95,
    temperature=1.75,
    num_return_sequences=count,
    pad_token_id=tokenizer.eos_token_id,
    num_beams=4,
    early_stopping=True,
  )

  return [tokenizer.decode(output, skip_special_tokens=True) for output in sample_outputs]

# Starting the app
app = Flask(__name__)

# Route definitions
@app.route('/descriptions/real')
def get_real_descriptions():
  count = int(request.args.get('count'))
  return random.sample(real_descriptions, count)

@app.route('/descriptions/fake')
def get_fake_descriptions():
  count = int(request.args.get('count'))
  return generate_fake_descriptions(count)

