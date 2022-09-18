create table 취미테이블(SubjectNo varchar2(20), Hobby varchar2(20));
select*from 취미테이블;
insert  into  취미테이블  values('S0001',' 테니스');
insert into  취미테이블  values('S0002', '탁구');
insert into 취미테이블 values('S0003', '볼링');

create table 운동날짜테이블(SubjectNo varchar2(20), 날짜 date);
insert into  운동날짜테이블 values('S0001','2019-01-01');
insert into  운동날짜테이블 values('S0001','2019-03-02');
insert into  운동날짜테이블 values('S0001','2019-04-20');
insert into  운동날짜테이블 values('S0001','2019-06-04');
insert into  운동날짜테이블 values('S0001','2019-07-15');
insert into  운동날짜테이블 values('S0001','2019-08-06');
select*from 운동날짜테이블;




