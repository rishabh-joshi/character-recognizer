import click

@click.command()
@click.argument('username')
@click.argument('password')
@click.argument('dbaddress')
@click.argument('database')

def writeCredential(username,password,dbaddress,database):
    file = open("config","w")
    file.write(username+","+password+","+dbaddress+","+database)
    print("credential updated succesfully!")

writeCredential()