from flask_table import Table, Col


class DataTable(Table):
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


class DataRow:
    """A class to store a row in the table in the database."""

    def __init__(self, row):
        """Constructor for dataRow

        Args:
            row (list): List id and character in one row.

        """
        self.id = row[0]
        self.character = row[1]


def get_html_table(query_result):
    """Generate the HTML code to display a table.

    Args:
        query_result (ResultProxy): A `ResultProxy` representing results of an SQL statement execution.

    Returns:
        str: The HTML code for the given table.

    """
    # fetch all the rows from the query
    result = []
    for query_row in query_result:
        result.append(query_row)

    # create DataRow objects from all the rows
    rows = [DataRow(row) for row in result]

    # generate the HTML code for the table
    table = DataTable(rows)
    return table