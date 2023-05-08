from flask import Flask, request 
import random

def load_real_descriptions():
  with open('real-descriptions.txt', "r", encoding="utf-8") as f:
    return [line.strip() for line in f.readlines()]

real_descriptions = load_real_descriptions()
app = Flask(__name__)

@app.route('/descriptions/real')
def get_real_descriptions():
  count = int(request.args.get('count'))
  return random.sample(real_descriptions, count)


