from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = dbAddress(get_credential()[0],get_credential()[1],get_credential()[2],get_credential()[3])
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config.from_envvar('APP_SETTINGS', silent=False)
db = SQLAlchemy(app)

