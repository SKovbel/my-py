import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col, concat_ws, lit

def main(session: snowpark.Session): 
    dataframe1 = session.table('snowflake_sample_data.TPCDS_SF100TCL.CUSTOMER')#.select("*").alias('CUSTOMER')
    dataframe2 = session.table('snowflake_sample_data.TPCDS_SF100TCL.CUSTOMER_ADDRESS')#.select("*").alias('ADDRESS')

    complex_query_result = session.create_dataframe(
        dataframe1
        .join(dataframe2, col('CA_ADDRESS_SK') == col('C_CURRENT_ADDR_SK'))
        .filter(col("C_SALUTATION") == 'Dr.')
        .select(col("C_CUSTOMER_SK").alias("Customer_Id"),
                col("C_FIRST_NAME").alias("Customer_First_Name"),
                col("C_LAST_NAME").alias("Customer_Last_Name"),
                concat_ws(lit(','),
                    col("CA_STREET_NUMBER"),
                    col("CA_SUITE_NUMBER"),
                    col("CA_CITY"),
                    col("CA_STATE"),
                    col("CA_ZIP"),
                    col("CA_COUNTRY")
                ).alias("Customer_Address"))
        .orderBy(col("C_FIRST_NAME").asc())
        .limit(100)
        .collect()
    )

    return complex_query_result
