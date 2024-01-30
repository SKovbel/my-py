CREATE FUNCTION dbo.CombineAverages
()
RETURNS DECIMAL(10, 2)
AS
BEGIN
    DECLARE @AvgSalary DECIMAL(10, 2);

    SELECT @AvgSalary = AVG(Salary)
    FROM Employees;

    RETURN @AvgSalary;
END;

GO;

CREATE AGGREGATE dbo.AverageSalary
(
    @input DECIMAL(10, 2) -- Input parameter type and size
)
RETURNS DECIMAL(10, 2)
-- Specify the combining function
WITH
(
    FUNCTION = 'dbo.CombineAverages', -- Combining function name
    INITCOND = 0
);
