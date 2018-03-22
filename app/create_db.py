from __init__ import db


def create_db():
    """Create the database table."""
    db.create_all()


class Prediction(db.Model):
    """The Prediction table which stores the predicted characters and their id.

    The table in the database would store the predicted values of the character 
    drawn by the user and the id. The id corresponds to how many characters
    have been predicted before this character in total.

    Attributes:
        id (Col): Unique ID column in the table.
        character (Col): Column storing the predicted user drawn character.

    """
    id = db.Column(db.Integer,primary_key=True)
    character = db.Column(db.String(1),unique=False,nullable=False)

    def __repr__(self):
        return '<Prediction Hist %r>'% id


if __name__ == '__main__':
    create_db()