def get_credential():
    file = open('db_credential.txt','r')
    credential = file.read()
    username = credential.split(",")[0]
    password = credential.split(",")[1]
    dbaddress = credential.split(",")[2]
    database = credential.split(",")[3]
    return [username,password,dbaddress,database]