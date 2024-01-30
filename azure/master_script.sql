-- EXEC('create_tables.sql');
-- EXEC('populate_tables.sql');

EXEC create_tables;
EXEC populate_tables;


IF OBJECT_ID('vEmployeeDepartments', 'V') IS NOT NULL
BEGIN
    DROP VIEW  vEmployeeDepartments;
    DROP VIEW  vEmployee;
END;

GO
CREATE OR ALTER VIEW vEmployeeDepartments AS
SELECT 
        e.EmployeeID,
        e.FirstName,
        e.LastName,
        e.BirthDate,
        e.HireDate,
        e.Salary,
        d.DepartmentID,
        d.DepartmentName
FROM Employees AS e
LEFT JOIN EmployeeDepartments AS ed ON ed.EmployeeID = e.EmployeeID
JOIN Departments AS d ON d.DepartmentID = ed.DepartmentID;


GO
CREATE OR ALTER VIEW vEmployee AS
SELECT 
        e.EmployeeID,
        e.FirstName,
        e.LastName,
        e.BirthDate,
        e.HireDate,
        e.Salary,
        COUNT(*) AS DepartmentCount,
        STRING_AGG(CONCAT(d.DepartmentID, CONCAT(':', REPLACE(d.DepartmentName, '|', '~~'))), '|') AS Departments
FROM Employees AS e
LEFT JOIN EmployeeDepartments AS ed ON ed.EmployeeID = e.EmployeeID
JOIN Departments AS d ON d.DepartmentID = ed.DepartmentID
GROUP BY e.EmployeeID,  e.FirstName, e.LastName, e.BirthDate, e.HireDate,  e.Salary;



