CREATE OR ALTER PROCEDURE create_tables
AS
BEGIN 
    DROP TABLE IF EXISTS Employees,
                     Departments,
                     EmployeeDepartments;

    CREATE TABLE Employees (
        EmployeeID INT PRIMARY KEY,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        BirthDate DATE,
        HireDate DATE,
        Salary DECIMAL(10, 2)
    );

    -- Create Departments table
    CREATE TABLE Departments (
        DepartmentID INT PRIMARY KEY,
        DepartmentName VARCHAR(100)
    );

    -- Create EmployeeDepartments table to establish a many-to-many relationship
    CREATE TABLE EmployeeDepartments (
        EmployeeID INT,
        DepartmentID INT,
        PRIMARY KEY (EmployeeID, DepartmentID),
        FOREIGN KEY (EmployeeID) REFERENCES Employees(EmployeeID),
        FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
    );

END;
