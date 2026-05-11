from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Flask is running</h1>"


if __name__ == '__main__':
    app.run(debug=True)
