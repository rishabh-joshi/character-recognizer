from init import db

def create_db():
    db.create_all()

class Prediction(db.Model):

    id = db.Column(db.Integer,primary_key=True)
    character = db.Column(db.String(1),unique=False,nullable=False)
    # area = db.Column(db.String(15),unique=False,nullable=False)
    # genre = db.Column(db.String(30),unique=False,nullable=False)
    # avg_temp = db.Column(db.Float,unique=False,nullable=False)
    # low_temp = db.Column(db.Float,unique=False,nullable=False)
    # precip = db.Column(db.Float,unique=False,nullable=False)
    # wind = db.Column(db.Float,unique=False,nullable=False)
    # sunlight = db.Column(db.Float,unique=False,nullable=False)
    # visitor_hist = db.Column(db.Integer,unique=False,nullable=False)
    # visitor_pred = db.Column(db.Integer,unique=False,nullable=False)

    def __repr__(self):
        return('<Prediction Hist %r>'% id)

if __name__ == '__main__':
    create_db()