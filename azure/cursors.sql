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
