create table if not exists public.Customers (
    CustomerID text primary key,
    CustomerName text not null,
    Branch text not null,
    City text,
    State text,
    Pincode text,
    Address text,
    Email text,
    Phone text,
    Balance numeric(12,2),
    CreditScore integer,
    AuthUserID uuid unique,
    CreatedAt timestamp default current_timestamp
);

create table if not exists public.LoanAccounts (
    LoanAccountID text primary key,
    CustomerID text not null references public.Customers(CustomerID),
    LoanType text,
    LoanAmount numeric(12,2),
    InterestRate numeric(5,2),
    TenureMonths integer,
    EMI numeric(12,2),
    StartDate date,
    EndDate date,
    OutstandingBalance numeric(12,2),
    LoanStatus text,
    CreatedAt timestamp default current_timestamp
);

create table if not exists public.Transactions (
    TransactionID text primary key,
    CustomerID text not null references public.Customers(CustomerID),
    TransactionDate timestamp not null,
    TransactionType text check (TransactionType in ('Credit','Debit')),
    Amount numeric(12,2),
    Merchant text,
    Category text,
    BalanceAfter numeric(12,2),
    CreatedAt timestamp default current_timestamp
);

create table if not exists public.UserRoles (
    user_id uuid primary key,
    role text check (role in ('customer','support','manager','risk','admin')),
    branch text,
    CreatedAt timestamp default current_timestamp
);

create table if not exists public.conversation_memory (
    id bigint generated always as identity primary key,
    user_id text not null,
    role text,
    query text not null,
    response text not null,
    route text,
    risk_level text,
    last_active_at timestamptz default now(),
    created_at timestamptz default now()
);

create index if not exists conversation_memory_user_id_idx on public.conversation_memory(user_id);
create index if not exists conversation_memory_last_active_at_idx on public.conversation_memory(last_active_at);

insert into public.Customers (CustomerID, CustomerName, Branch, City, State, Address, Email, Phone, Balance, CreditScore, AuthUserID) values
('C001','Asha Menon','Chennai Main','Chennai','Tamil Nadu','12 Lake Road','asha@example.com','9000000001',145000.00,742,'11111111-1111-1111-1111-111111111001'),
('C002','Rahul Singh','Chennai Main','Chennai','Tamil Nadu','9 North Street','rahul@example.com','9000000002',98000.00,701,'11111111-1111-1111-1111-111111111002'),
('C003','Priya Nair','Chennai Main','Chennai','Tamil Nadu','88 Green Avenue','priya@example.com','9000000003',210500.00,768,'11111111-1111-1111-1111-111111111003'),
('C004','Arjun Das','Chennai Main','Chennai','Tamil Nadu','5 Temple View','arjun@example.com','9000000004',65000.00,688,'11111111-1111-1111-1111-111111111004'),
('C005','Neha Kapoor','Chennai Main','Chennai','Tamil Nadu','41 River Lane','neha@example.com','9000000005',302000.00,791,'11111111-1111-1111-1111-111111111005'),
('C006','Vikram Rao','Chennai Main','Chennai','Tamil Nadu','21 Market Road','vikram@example.com','9000000006',87000.00,715,'11111111-1111-1111-1111-111111111006'),
('C007','Kiran Patel','Chennai Main','Chennai','Tamil Nadu','17 Palm Street','kiran@example.com','9000000007',154000.00,732,'11111111-1111-1111-1111-111111111007'),
('C008','Meera Iyer','Chennai Main','Chennai','Tamil Nadu','32 Sunrise Nagar','meera@example.com','9000000008',119000.00,709,'11111111-1111-1111-1111-111111111008'),
('C009','Sanjay Kumar','Chennai Main','Chennai','Tamil Nadu','77 Station Road','sanjay@example.com','9000000009',267000.00,774,'11111111-1111-1111-1111-111111111009'),
('C010','Anita Roy','Chennai Main','Chennai','Tamil Nadu','6 Garden Colony','anita@example.com','9000000010',134500.00,721,'11111111-1111-1111-1111-111111111010')
on conflict (CustomerID) do nothing;

insert into public.LoanAccounts (LoanAccountID, CustomerID, LoanType, LoanAmount, InterestRate, TenureMonths, EMI, StartDate, EndDate, OutstandingBalance, LoanStatus) values
('L001','C001','Home',2500000.00,8.50,240,21696.00,'2024-01-01','2044-01-01',2380000.00,'Active'),
('L002','C003','Personal',300000.00,11.25,48,7825.00,'2025-02-01','2029-02-01',255000.00,'Active'),
('L003','C005','Gold',150000.00,9.75,24,6900.00,'2025-05-01','2027-05-01',102000.00,'Active'),
('L004','C007','Auto',650000.00,9.10,60,13480.00,'2023-08-01','2028-08-01',388000.00,'Active'),
('L005','C009','Housing',1800000.00,8.80,180,17920.00,'2022-06-01','2037-06-01',1325000.00,'Active')
on conflict (LoanAccountID) do nothing;

insert into public.Transactions (TransactionID, CustomerID, TransactionDate, TransactionType, Amount, Merchant, Category, BalanceAfter) values
('T001','C001','2026-04-20 10:00:00','Credit',85000.00,'Employer','Salary',145000.00),
('T002','C001','2026-04-21 09:00:00','Debit',1200.00,'Metro','Travel',143800.00),
('T003','C001','2026-04-22 14:00:00','Debit',2500.00,'Amazon','Shopping',141300.00),
('T004','C001','2026-04-23 11:00:00','Debit',21696.00,'Loan EMI','EMI',119604.00),
('T005','C001','2026-04-24 18:00:00','Credit',25400.00,'Freelance','Income',145004.00),
('T006','C002','2026-04-20 10:00:00','Credit',60000.00,'Employer','Salary',98000.00),
('T007','C002','2026-04-21 08:30:00','Debit',1800.00,'Swiggy','Food',96200.00),
('T008','C002','2026-04-22 13:15:00','Debit',3200.00,'DMart','Groceries',93000.00),
('T009','C002','2026-04-23 17:45:00','Debit',4500.00,'Hospital','Medical',88500.00),
('T010','C002','2026-04-24 20:20:00','Credit',9500.00,'Refund','Credit',98000.00),
('T011','C003','2026-04-20 10:00:00','Credit',110000.00,'Employer','Salary',210500.00),
('T012','C003','2026-04-21 09:45:00','Debit',7825.00,'Loan EMI','EMI',202675.00),
('T013','C003','2026-04-22 12:10:00','Debit',6400.00,'IKEA','Shopping',196275.00),
('T014','C003','2026-04-23 15:30:00','Debit',2100.00,'Cafe','Food',194175.00),
('T015','C003','2026-04-24 19:05:00','Credit',16325.00,'Transfer In','Credit',210500.00),
('T016','C004','2026-04-20 10:00:00','Credit',42000.00,'Employer','Salary',65000.00),
('T017','C004','2026-04-21 08:00:00','Debit',850.00,'Bus Pass','Travel',64150.00),
('T018','C004','2026-04-22 16:00:00','Debit',1900.00,'Grocery Mart','Groceries',62250.00),
('T019','C004','2026-04-23 18:00:00','Debit',4500.00,'Rent','Bills',57750.00),
('T020','C004','2026-04-24 21:00:00','Credit',7250.00,'Friend','Credit',65000.00),
('T021','C005','2026-04-20 10:00:00','Credit',150000.00,'Employer','Salary',302000.00),
('T022','C005','2026-04-21 11:00:00','Debit',6900.00,'Loan EMI','EMI',295100.00),
('T023','C005','2026-04-22 14:30:00','Debit',15000.00,'Jewellery','Shopping',280100.00),
('T024','C005','2026-04-23 17:10:00','Debit',3000.00,'Dining','Food',277100.00),
('T025','C005','2026-04-24 20:30:00','Credit',24900.00,'Dividend','Income',302000.00)
on conflict (TransactionID) do nothing;

insert into public.UserRoles (user_id, role, branch) values
('11111111-1111-1111-1111-111111111001','customer','Chennai Main'),
('11111111-1111-1111-1111-111111111002','customer','Chennai Main'),
('11111111-1111-1111-1111-111111111003','customer','Chennai Main'),
('11111111-1111-1111-1111-111111111004','customer','Chennai Main'),
('11111111-1111-1111-1111-111111111005','customer','Chennai Main'),
('11111111-1111-1111-1111-111111111006','customer','Chennai Main'),
('11111111-1111-1111-1111-111111111007','customer','Chennai Main'),
('11111111-1111-1111-1111-111111111008','customer','Chennai Main'),
('11111111-1111-1111-1111-111111111009','customer','Chennai Main'),
('11111111-1111-1111-1111-111111111010','customer','Chennai Main'),
('22222222-2222-2222-2222-222222222001','manager','Chennai Main'),
('22222222-2222-2222-2222-222222222002','support',null),
('22222222-2222-2222-2222-222222222003','risk',null),
('22222222-2222-2222-2222-222222222004','admin',null)
on conflict (user_id) do nothing;
