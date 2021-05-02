from flask import Blueprint, render_template

# from .models import Data
# from . import db

views = Blueprint("views", __name__)


@views.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")


@views.route("/projects")
def projects():
    return render_template("projects.html")


@views.route("/about")
def about():
    return render_template("about.html")

@views.route("/game")
def game():
    return render_template("game.html")
