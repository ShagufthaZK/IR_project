from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quizgen')
def quizgen():
    return render_template('quizgen.html')