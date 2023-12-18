import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col

def main(session: snowpark.Session): 
    # Your code goes here, inside the "main" handler.
    table1_name = 'snowflake_sample_data.TPCDS_SF100TCL.CUSTOMER'  # Replace with your actual table name
    table2_name = 'snowflake_sample_data.TPCDS_SF100TCL.CUSTOMER_ADRESS'  # Replace with your actual table name
    join_column = 'common_column'  # Replace with the column you want to join on

    # Create DataFrames for the specified tables
    dataframe1 = session.table(table1_name)
    dataframe2 = session.table(table2_name)

    # Complex SQL query with joins, filters, and selections
    complex_query_result = (
        dataframe1
        .join(dataframe2, col(join_column))
        .filter(col("some_column") > 100)
        .select(col("table1_column"), col("table2_column"))
        .orderBy(col("table1_column").asc())
        .limit(100)
        .execute()
    )
C_CURRENT_ADDR_SK

    # Print the query result
    complex_query_result.show()

    # Return the result (optional, for display in the Results tab)
    return complex_query_result