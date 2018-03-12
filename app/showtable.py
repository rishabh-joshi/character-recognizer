from flask_table import Table, Col

class dataTable(Table):
    id = Col("id")
    character = Col('character')

class dataRow:
    def __init__(self, row):
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
