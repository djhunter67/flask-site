from flask import Blueprint, render_template, request
from game.game_tab import game_file, dev, info, time_to_calc

# from .models import Data
# from . import db

views = Blueprint("views", __name__)


@views.route("/", methods=["GET", "POST"])
def home():
    return render_template("home/home.html")


@views.route("/projects")
def projects():
    return render_template("projects/projects.html")


@views.route("/about")
def about():
    return render_template("about/about.html")


@views.route("/game", methods=["GET", "POST"])
def game():
    from torch import from_numpy
    from numpy import array

    if request.method == "GET":
        return render_template("game/game.html")
    elif request.method == "POST":
        try:
            input_val = int(request.form["value"])
            if input_val > 50:
                return render_template(
                    "game/game.html", game_holder="ENTER AN INTEGER LESS THAN 51", info=""
                )
            torch_val = from_numpy(array(input_val)).to(dev)
            val = game_file(torch_val.item())
            return render_template(
                "game/game.html", game_holder=f"{val:,}", info=info, number=input_val, time=time_to_calc)
        except:
            return render_template(
                "game/game.html", game_holder="INVALID INPUT", info=""
            )
