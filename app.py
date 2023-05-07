from flask import Flask, request 

app = Flask(__name__)

@app.route('/descriptions')
def index():
  count = int(request.args.get('count'))

  return ['Random description' for i in range(count)]