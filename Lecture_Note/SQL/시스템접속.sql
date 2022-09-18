show user;

create user nice identified by 123456;
grant connect, resource to nice;
conn nice;