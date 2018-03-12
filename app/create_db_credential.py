import click

@click.command()
@click.argument('username')
@click.argument('password')
@click.argument('dbaddress')
@click.argument('database')

def writeCredential(username,password,dbaddress,database):
    file = open("db_credential.txt","w")
    file.write(username+","+password+","+dbaddress+","+database)
    print("credential updated succesfully!")

writeCredential()