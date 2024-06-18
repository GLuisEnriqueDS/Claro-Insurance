
import pandas as pd
import pymysql
import logging
import sshtunnel
from sshtunnel import SSHTunnelForwarder

ssh_host = ''
ssh_username = ''
ssh_password = ''
database_username = ''   
database_password = ''
database_name = 'salesforce_datalake'
localhost = ''

def myblacklist():
    open_ssh_tunnel()
    mysql_connect()
    df_tunel = run_query("SELECT * FROM sfs_books_of_business")
    mysql_disconnect()
    close_ssh_tunnel()
    return df_tunel

def open_ssh_tunnel(verbose=False):
    """Open an SSH tunnel and connect using a username and password.
    
    :param verbose: Set to True to show logging
    :return tunnel: Global SSH tunnel connection
    """
    
    if verbose:
        sshtunnel.DEFAULT_LOGLEVEL = logging.DEBUG
    
    global tunnel
    tunnel = SSHTunnelForwarder(
        (ssh_host, 22),
        ssh_username = ssh_username,
        ssh_password = ssh_password,
        remote_bind_address = ('127.0.0.1', 2206)
    )
            
    tunnel.start()


def mysql_connect():
    global connection
        
    connection = pymysql.connect(
        host='127.0.0.1',
        user=database_username,
        passwd=database_password,
        db=database_name,
        port=tunnel.local_bind_port
        )

def run_query(sql):
    return pd.read_sql_query(sql, connection)

def mysql_disconnect():    
    connection.close()

def close_ssh_tunnel():   
    tunnel.close

    