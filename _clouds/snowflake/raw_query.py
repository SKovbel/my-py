# add package pyarrow
import snowflake.snowpark as snowpark

def main(session: snowpark.Session): 
    raw_sql_query = '''
    SELECT C_FIRST_NAME, C_LAST_NAME
    FROM snowflake_sample_data.TPCDS_SF100TCL.CUSTOMER
    WHERE C_SALUTATION = 'Dr.'
    ORDER BY C_FIRST_NAME ASC
    LIMIT 100
    '''

    result = session.sql(raw_sql_query)
    return result
