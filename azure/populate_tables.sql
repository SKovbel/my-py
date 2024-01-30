CREATE OR ALTER PROCEDURE populate_tables
AS
BEGIN 
    INSERT INTO Employees (EmployeeID, FirstName, LastName, BirthDate, HireDate, Salary)
    VALUES
        (1, 'John', 'Doe', '1990-01-15', '2015-03-20', 60000.00),
        (2, 'Jane', 'Smith', '1985-08-22', '2018-06-10', 75000.00),
        (3, 'Bob', 'Johnson', '1992-05-05', '2017-01-08', 55000.00),
        (4, 'Alice', 'Brown', '1988-11-10', '2016-09-14', 70000.00);

    -- Insert sample data into Departments table
    INSERT INTO Departments (DepartmentID, DepartmentName)
    VALUES
        (1, 'IT'),
        (2, 'HR'),
        (3, 'Finance');

    -- Insert sample data into EmployeeDepartments table
    INSERT INTO EmployeeDepartments (EmployeeID, DepartmentID)
    VALUES
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 1),
        (4, 3);
END;
