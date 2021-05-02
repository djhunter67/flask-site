from flask import Flask
#from flask_sqlalchemy import SQLAlchemy


# db = SQLAlchemy()
# DB_NAME = "database.db"
KEYS = "key_file.txt"

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = KEYS

    from .views import views

    app.register_blueprint(views, url_prefix="/")

    return app


"""
def create_db(app):
    if not path.exists("flask_prac/" + DB_NAME):
        db.create_all(app=app)
        print("Created Database!")
"""
