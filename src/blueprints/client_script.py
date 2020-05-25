from flask import Blueprint, send_file


client_script = Blueprint("client_script", __name__)


@client_script.route("/index", methods=["GET", "POST"])
def show_widget_script():
    return send_file("app/index.js")