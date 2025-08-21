create table staff (
id serial PRIMARY KEY,
name VARCHAR(50) NOT NULL,
age INT NOT NULL
)

drop table if exists staff
select*from staff