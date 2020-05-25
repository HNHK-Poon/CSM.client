# import sys
# print(sys.path)
# sys.path.remove('/home/poon/PycharmProjects/CSM.client/src/testing')
# sys.path.extend(['/home/poon/PycharmProjects/CSM.client/src'])
from config.ConfigValue import ConfigValue
from flask import Flask
from blueprints.client_blueprint import client_blueprint
from blueprints.client_script import client_script
from flask_cors import CORS


host = ConfigValue().get_value(ConfigValue().GENERAL, "Host")
port = ConfigValue().get_value(ConfigValue().GENERAL, "Port")

app = Flask(__name__, template_folder="app", static_folder="app")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)


app.register_blueprint(client_script)
app.register_blueprint(client_blueprint)

if __name__ == "__main__":
    app.run(host, port, debug=False)