CREATE OR ALTER FUNCTION CalculateTotalSalary
(
    @HourlyRate DECIMAL(10, 2),
    @HoursWorked INT
)
RETURNS DECIMAL(10, 2)
AS
BEGIN
    DECLARE @TotalSalary DECIMAL(10, 2);

    -- Calculate total salary
    SET @TotalSalary = @HourlyRate * @HoursWorked;

    -- Return the result
    RETURN @TotalSalary;
END;


/*
DECLARE @HoursWorked INT = 40;

SELECT dbo.CalculateTotalSalary(@HourlyRate, @HoursWorked) AS TotalSalary;
SELECT FirstName, LastName, Salary, dbo.CalculateTotalSalary(Salary, @HoursWorked) FROM dbo.GetEmployeeInfo2(500, 1500, 'John');
*/