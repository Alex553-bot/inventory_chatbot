import psycopg2
from psycopg2 import sql

from dotenv import load_dotenv
import os

import pandas as pd

load_dotenv()

def get_results(query):
    """
    Executes the query and returns the result in a DataFrame

    Params:
    query (str): SQL query to be executed

    Return:
    (pandas.DataFrame) results
    """
    try:
        db_params = {
            'dbname':   os.getenv('DB_NAME'),
            'user':     os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host':     os.getenv('DB_HOST'),
            'port':     os.getenv('DB_PORT')
        }
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        cursor.execute(query)

        records = cursor.fetchall()
        cols = [des[0] for des in cursor.description]
        
        df = pd.DataFrame(records, columns=cols)

    except Exception as e: 
        print(e)
        df = pd.DataFrame()
    finally:
        conn.close()
        cursor.close()
        return df