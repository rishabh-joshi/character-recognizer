from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# create the application object
app = Flask(__name__)

# configure the database settings from the environment variable
app.config.from_envvar('APP_SETTINGS', silent=False)

# create the database instance from the application object
db = SQLAlchemy(app)