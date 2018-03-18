from flask_table import Table, Col

class dataTable(Table):
    """A class to store the entries in a row of a database table.
	
	The table in the database would store the predicted values of the
	character drawn by the user. This class stores the entries in one
	row of that table. This class inherits from the Table class in
	flask_table module.

    Attributes:
        id (Col): Unique ID column in the table.
        character (Col): Column storing the predicted user drawn character.

    """
    id = Col("id")
    character = Col('character')

class dataRow:
    """A class to store a row in the table in the database.

    Attributes:
        id (str): Unique ID for each row in the table.
        character (str): The predicted character drawn by the user.

    """
    def __init__(self, row):
        """Example of docstring on the __init__ method.

        Args:
            row (list): List id and character in one row.

        """
        self.id = row[0]
        self.character = row[1]

def getHtmlTable(queryResult):
    result = []
    for queryRow in queryResult:
        result.append(queryRow)
    rows = [dataRow(row) for row in result]
    table = dataTable(rows)
    # print(table.__html__())
    return table
