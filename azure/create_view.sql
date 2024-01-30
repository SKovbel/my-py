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
JOIN Departments AS d ON d.DepartmentID = ed.DepartmentID


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
GROUP BY e.EmployeeID,  e.FirstName, e.LastName, e.BirthDate, e.HireDate,  e.Salary
