"""very basic web application, which serves static resources and provides an API to edit task lists"""
from flask import Flask
from flask import request
import os

app = Flask(__name__)


@app.route("/")
def index():
    return open("index.html").read()


@app.route("/small")
def small():
    return open("index-small.html").read()


@app.route("/data/<file>", methods=['GET', 'POST'])
def data(file):
    if request.method == "GET":
        return open(os.path.join("data", file)).read()
    elif request.method == "POST":
        print(request.form)
        new_data = request.form.get("data")
        open(os.path.join("data", file), "w").write(new_data)
        return "{'success': true}"
