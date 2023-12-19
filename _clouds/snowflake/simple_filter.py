import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col

def main(session: snowpark.Session): 
    # Your code goes here, inside the "main" handler.
    tableName = 'snowflake_sample_data.TPCDS_SF100TCL.CUSTOMER'
    dataframe = session.table(tableName).filter(col("C_SALUTATION") == 'Dr.')
    new_df = dataframe.limit(100)
    return new_df
