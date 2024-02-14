CREATE OR ALTER PROCEDURE test_cursor
    @ret_cur CURSOR VARYING OUTPUT
AS
BEGIN 
    SET NOCOUNT ON;

    DECLARE @CURx CURSOR;
    SET @CURx = CURSOR FOR SELECT * FROM Employees;

    SET @ret_cur = @CURx;
    /*
        DECLARE @EmployeeID INT, @FirstName NVARCHAR(50), @LastName NVARCHAR(50);
        EXEC test_cursor @ret_cur = @EmpCursor OUTPUT;
        OPEN @return_cursor;
        FETCH NEXT FROM @EmpCursor INTO @EmployeeID, @FirstName, @LastName;
        WHILE @@FETCH_STATUS = 0
        BEGIN
            PRINT 'EmployeeID: ' + CAST(@EmployeeID AS NVARCHAR(10)) +
                ', FirstName: ' + @FirstName +
                ', LastName: ' + @LastName;

            -- Fetch the next row
            FETCH NEXT FROM @EmpCursor INTO @EmployeeID, @FirstName, @LastName;
        END
        CLOSE @EmpCursor;
        DEALLOCATE @EmpCursor;
    */
END;



Hi Eva, 
sorry for delay.

My Java experience mostly JEE/Jakarta. One Java-mob and 3 Kotlin-mob projects.
Also were 2 scripts on AWS Scala (just for interesting how it close to Java)
Except Kotlin projects (which used API and API used DB) all Java-projects were high level projects around databases.
No threads, parallelism I did with Queue and MDB beans. No Graphics, mostly form/grid interfaces + media.
If it interesting for you we can talk.

