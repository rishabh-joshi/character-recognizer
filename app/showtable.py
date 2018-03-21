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
    """A class to store a row in the table in the database."""
    def __init__(self, row):
        """Constructor for dataRow

        Args:
            row (list): List id and character in one row.

        """
        self.id = row[0]
        self.character = row[1]

def getHtmlTable(queryResult):
    """Generate the HTML code to display a table.
    
    Args:
        queryResult (ResultProxy): A `ResultProxy` representing results of an SQL statement execution.

    Returns:
        str: The HTML code for the given table.

    """
    result = []
    for queryRow in queryResult:
        result.append(queryRow)
    rows = [dataRow(row) for row in result]
    table = dataTable(rows)
    # print(table.__html__())
    return table
