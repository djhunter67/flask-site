from flask import Blueprint, render_template, request
from torch import tensor

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
    from torch import cuda, device, LongTensor, from_numpy
    from numpy import array
    GPU = "0"  # 0: 2080TI;  1: P2000
    dev = device("cuda:" + GPU if cuda.is_available() else "cpu")
    info = f"Executing via: {dev}"
   
    def game_file(n):
        # Recursive Fibonacci as a placeholder for a game
        #
        if n <= 1:
            return n
        else:
            return (game_file(n - 1) + game_file(n - 2))
    
    if request.method == "GET":
        return render_template("game/game.html")
    elif request.method == "POST":
        try:
            torch_val = from_numpy(array(int(request.form["value"]))).to(dev)
            val = game_file(torch_val.item())
            return render_template("game/game.html", game_holder=f"{val:,}", info=info)
        except:
            return render_template("game/game.html", game_holder="INVALID INPUT", info="")