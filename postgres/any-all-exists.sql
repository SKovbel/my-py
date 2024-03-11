SELECT 
  * 
FROM 
  employees 
WHERE 
  salary = ANY (
    SELECT 
      salary 
    FROM 
      managers
  );

-- ALL
SELECT 
  * 
FROM 
  employees 
WHERE 
  salary > ALL(
    select 
      salary 
    from 
      managers
  );

  /*
value > ALL (subquery) returns true if the value is greater than the biggest value returned by the subquery.
value >= ALL (subquery) returns true if the value is greater than or equal to the biggest value returned by the subquery.
value < ALL (subquery) returns true if the value is less than the smallest value returned by the subquery.
value <= ALL (subquery) returns true if the value is less than or equal to the smallest value returned by the subquery.
value = ALL (subquery) returns true if the value is equal to every value returned by the subquery.
value != ALL (subquery) returns true if the value is not equal to any value returned by the subquery.
  */