GO
CREATE OR ALTER FUNCTION GetEmployeeInfo1(
    @MinSalary INT = 0,
    @MaxSalary INT = 1000,
    @Name NVARCHAR(50) = '%'
)
RETURNS TABLE
AS
RETURN
(
    -- Query to select data
    SELECT EmployeeID, FirstName, LastName
    FROM Employees
    WHERE FirstName LIKE @Name OR 
          (Salary >= @MinSalary AND Salary <= @MaxSalary)
);

GO
CREATE OR ALTER FUNCTION GetEmployeeInfo2(
    @MinSalary INT = 0,
    @MaxSalary INT = 1000,
    @Name NVARCHAR(50) = '%'
)
RETURNS @ResultTable TABLE (
    EmployeeID INT,
    FirstName NVARCHAR(50),
    LastName NVARCHAR(50),
    Salary DECIMAL(10, 2)
)
AS
BEGIN
    DECLARE @EmployeeID INT, @FirstName NVARCHAR(50), @LastName NVARCHAR(50), @AdjustedSalary INT;

    DECLARE EmployeeCursor CURSOR FOR
    SELECT EmployeeID, FirstName, LastName, Salary
    FROM Employees;

    OPEN EmployeeCursor;

    FETCH NEXT FROM EmployeeCursor INTO @EmployeeID, @FirstName, @LastName, @AdjustedSalary;

    WHILE @@FETCH_STATUS = 0
    BEGIN
        -- Your loop logic or calculations go here
        -- For example, you can add 100 to Salary
        SET @AdjustedSalary = @AdjustedSalary + 100;

        -- Check conditions and insert into the result table if needed
        IF @FirstName LIKE @Name OR (@AdjustedSalary >= @MinSalary AND @AdjustedSalary <= @MaxSalary)
        BEGIN
            INSERT INTO @ResultTable (EmployeeID, FirstName, LastName, Salary)
            VALUES (@EmployeeID, @FirstName, @LastName, @AdjustedSalary);
        END

        FETCH NEXT FROM EmployeeCursor INTO @EmployeeID, @FirstName, @LastName, @AdjustedSalary;
    END

    CLOSE EmployeeCursor;
    DEALLOCATE EmployeeCursor;

    RETURN;
END;


/*
SELECT * FROM GetEmployeeInfo1(500, 1500, 'John');
SELECT * FROM dbo.GetEmployeeInfo2(500, 1500, 'John');
*/




