show user;
select*from countries;

create table sawon(
name varchar2(20),
grade varchar2(10),
job number
);

select*from sawon;--������̺��� ������ ���
desc sawon;

--data �߰�
insert into sawon values(
'hong', 'manager', '10');

select*from sawon;

insert into sawon values(
'yunyoseob', 'manager', '10');

select*from sawon;

insert into sawon values(
'kimyongoak', 'stupid', '10');

select*from sawon;

insert into sawon values(
'lee', 'manager', '10');

select*from sawon;

insert into sawon values(
'jung', 'manager', '10');

select*from sawon;

insert into sawon values(
'no', 'manager', '10');

insert into sawon values(
'park', 'ceo', '20');

select*from sawon;

insert into sawon(name, grade) values ('hoon', 'marketing');

select*from sawon;

--sawon2 ���̺� �����
create table sawon2(
name varchar2(20),
grade varchar2(20),
ibsdata date default sysdate);
desc sawon2;
insert into sawon2
values('song','mgr',sysdate);
select*from sawon2;

insert into sawon2
values('kim', 'sls', sysdate);
select*from sawon2;

insert into sawon2
values('jin','ppr',sysdate);
select*from sawon2;

insert into sawon2(name, grade) values('chik','sales');

insert into sawon2 values('yoon','mgr','21/03/12');
select*from sawon2;
commit;

select*from sawon2;

create table sawon3(
empno number(12), ename varchar2(12), job varchar2(12), 
mgr number(12), hiredate date, sal number(12), comm number(12),
deptno number(12));

insert into sawon3(empno, ename, job, mgr, hiredate, sal, deptno) values
('7369','SMITH','CLERK','7902','80-12-17','800','20');

select*from sawon3;

insert into sawon3 values('7499','ALLEN','SALESMAN','7698','81-02-20','1600'
,'300','30');

select*from sawon3;

insert into sawon3 values('7521','WARD','SALESMAN','7698','81-02-22','1250','500','30');

select*from sawon3;

insert into sawon3(empno, ename, job, mgr, hiredate, sal, deptno) values
('7566','JONES','MANAGER','7839','81-04-02','2975','20');

select*from sawon3;

insert into sawon3 values('7654','MARTIN','SALESMAN','7698','81-09-28','1250','1400','30');

select*from sawon3;

insert into sawon3(empno, ename, job, mgr, hiredate, sal, deptno) values
('7698','BLAKE','MANAGER','7839','81-05-01','2850','30');

select*from sawon3;

insert into sawon3(empno, ename, job, mgr, hiredate, sal, deptno) values
('7782','CLARK','MANAGER','7839','81-06-09','2450','10');

select*from sawon3;

insert into sawon3(empno, ename, job, hiredate, sal, deptno) values
('7839','KING','PRESIDENT','81-11-17','5000','10');

select*from sawon3;

insert into sawon3 values('7844','TURNER','SALESMAN','7698','81-09-08','1500','0','30');

select*from sawon3;

insert into sawon3(empno, ename, job, mgr, hiredate, sal, deptno) values
('7900',);

drop table sawon3;

create table dept(
deptno number(4), dname varchar2(20), loc varchar2(20));

insert into dept values(10,'ACCOUNTING','NEW YORK');

insert into dept values(20,'RESEARCH','DALLAS');

insert into dept values(30,'SALES','CHICAGO');

select*from dept;

insert into dept values(40,'OPERATIONS','BOSTON');

select*from dept;

drop table dept;

drop table sawon;
drop table sawon2;

select*from tab;
show recyclebin;
purge recyclebin;
drop table sawon purge;

select*from tab;

create table sawon(
sano number primary key, saname varchar2(20)not null,
saaddr varchar2(100));

insert into sawon values(10, 'hong1', 'seoul');
insert into sawon values(20, 'hong2', 'pusan');
insert into sawon values(30, 'hong3', 'daegu');
commit;
select*from sawon;
insert into sawon values(20, 'hong4', 'seoul');
--sano �Ӽ��� primary key �������� ������
insert into sawon values(40, null, 'incheon');
--saname �Ӽ��� not null �������� ������

CREATE TABLE DEPT
       (DEPTNO number(10),
        DNAME VARCHAR2(14),
        LOC VARCHAR2(13) );



INSERT INTO DEPT VALUES (10, 'ACCOUNTING', 'NEW YORK');
INSERT INTO DEPT VALUES (20, 'RESEARCH',   'DALLAS');
INSERT INTO DEPT VALUES (30, 'SALES',      'CHICAGO');
INSERT INTO DEPT VALUES (40, 'OPERATIONS', 'BOSTON');
select*from dept;

CREATE TABLE EMP (
 EMPNO               NUMBER(4) NOT NULL,
 ENAME               VARCHAR2(10),
 JOB                 VARCHAR2(9),
 MGR                 NUMBER(4) ,
 HIREDATE            DATE,
 SAL                 NUMBER(7,2),
 COMM                NUMBER(7,2),
 DEPTNO              NUMBER(2) );



INSERT INTO EMP VALUES (7839,'KING','PRESIDENT',NULL,'81-11-17',5000,NULL,10);
INSERT INTO EMP VALUES (7698,'BLAKE','MANAGER',7839,'81-05-01',2850,NULL,30);
INSERT INTO EMP VALUES (7782,'CLARK','MANAGER',7839,'81-05-09',2450,NULL,10);
INSERT INTO EMP VALUES (7566,'JONES','MANAGER',7839,'81-04-01',2975,NULL,20);
INSERT INTO EMP VALUES (7654,'MARTIN','SALESMAN',7698,'81-09-10',1250,1400,30);
INSERT INTO EMP VALUES (7499,'ALLEN','SALESMAN',7698,'81-02-11',1600,300,30);
INSERT INTO EMP VALUES (7844,'TURNER','SALESMAN',7698,'81-08-21',1500,0,30);
INSERT INTO EMP VALUES (7900,'JAMES','CLERK',7698,'81-12-11',950,NULL,30);
INSERT INTO EMP VALUES (7521,'WARD','SALESMAN',7698,'81-02-23',1250,500,30);
INSERT INTO EMP VALUES (7902,'FORD','ANALYST',7566,'81-12-11',3000,NULL,20);
INSERT INTO EMP VALUES (7369,'SMITH','CLERK',7902,'80-12-09',800,NULL,20);
INSERT INTO EMP VALUES (7788,'SCOTT','ANALYST',7566,'82-12-22',3000,NULL,20);
INSERT INTO EMP VALUES (7876,'ADAMS','CLERK',7788,'83-01-15',1100,NULL,20);
INSERT INTO EMP VALUES (7934,'MILLER','CLERK',7782,'82-01-11',1300,NULL,10);


@c:\empdept.sql;

select*from EMP;
--���̺��� �Ϻ� ������ ���
select empno, job, hiredate from emp;
select empno, job, to_char(hiredate,'yyyy-nn-dd') from emp;
--�ٸ� ������ ��¥ �Է½� ������ ���� �ذ�

INSERT INTO EMP VALUES (
8934,'MILLER','CLERK',7782,TO_DATE('1982-01-11','yyyy-mm-dd'), 1300,NULL,10);
 
 drop table emp;
 drop table professor ;

show user;
@c:\professor.sql;

create table professor
(profno number(4) primary key,
 name  varchar2(20) not null, 
 id  varchar2(15) not null,
 position varchar2 (30) not null,
 pay number (3) not null,
 hiredate  date not null,
 bonus number(4) ,
 deptno  number(3),
 email  varchar2(50),
 hpage  varchar2(50)) tablespace users;

insert into professor
values(1001,'Audie Murphy','Murphy','a full professor',550,to_date('1980-06-23','YYYY-MM-DD'),
100,101,'captain@abc.net','http://www.abc.net');

insert into professor
values(1002,'Angela Bassett','Bassett','assistant professor',380,to_date('1987-01-30','YYYY-MM-DD'),
60,101,'sweety@abc.net','http://www.abc.net');

insert into professor
values (1003,'Jessica Lange','Lange','instructor',270,
to_date('1998-03-22','YYYY-MM-DD'),null,101,'pman@power.com','http://www.power.com');

insert into professor
values (2001,'Winona Ryder','Ryder','instructor',250,to_date('2001-09-01','YYYY-MM-DD'),
null,102,'lamb1@hamail.net',null);

insert into professor
values (2002,'Michelle Pfeiffer','Pfeiffer','assistant professor',350,to_date('1985-11-30','YYYY-MM-DD'),
80,102,'number1@naver.com','http://num1.naver.com');

insert into professor
values (2003,'Whoopi Goldberg','Goldberg','a full professor',490,to_date('1982-04-29','YYYY-MM-DD'),90,
102,'bdragon@naver.com',null);

insert into professor
values (3001,'Emma Thompson','Thompson','a full professor',530,to_date('1981-10-23','YYYY-MM-DD'),
110,103,'angel1004@hanmir.com',null);

insert into professor
values (3002,'Julia Roberts','Roberts','assistant professor',330,to_date('1997-07-01','YYYY-MM-DD'),
50,103,'naone10@empal.com',null);

insert into professor
values (3003,'Sharon Stone','Stone','instructor',290,to_date('2002-02-24','YYYY-MM-DD'),null,
103,'only_u@abc.com',null);

insert into professor
values (4001,'Meryl Streep','Streep','a full professor',570,to_date('1981-10-23','YYYY-MM-DD'),
130,201,'chebin@daum.net',null);

insert into professor
values (4002,'Susan Sarandon','Sarandon','assistant professor',330,to_date('2009-08-30','YYYY-MM-DD'),
null,201,'gogogo@def.com',null);

insert into professor
values (4003,'Nicole Kidman','Kidman','assistant professor',310,to_date('1999-12-01','YYYY-MM-DD'),50,202,
'mypride@hanmail.net',null);

insert into professor
values (4004,'Holly Hunter','Hunter','instructor',260,to_date('2009-01-28','YYYY-MM-DD'),null,202,
'ironman@naver.com',null);

insert into professor
values (4005,'Meg Ryan','Ryan','a full professor',500,to_date('1985-09-18','YYYY-MM-DD'),80,203,
'standkang@naver.com',null);

insert into professor 
values (4006,'Andie Macdowell','Macdowell','instructor',220,to_date('2010-06-28','YYYY-MM-DD'),null,
301,'napeople@jass.com',null);

insert into professor
values (4007,'Jodie Foster','Foster','assistant professor',290,to_date('2001-05-23','YYYY-MM-DD'),30,301,
'silver-her@daum.net',null);

commit;

select*from professor;
--name �ʵ� ���
select name from professor;
--�ʵ� �̿� �������
select name, 'good morng'"Good Morning"from professor;
--��Ī(Alias)*****
select*from professor;
select profno as "������ȣ", name as "�����̸�",
id as "�������̵�",
position as "��������",
pay as "�޿�",
email as "�̸���" from professor;
select*from professor;

--���ʽ� �λ�
select profno, name, id, position, pay*1.1 from professor;
select profno, name, id, position, pay*1.1 as "�λ�ݾ�10%" from professor;

select profno ������ȣ,
name as "�����̸�" from professor;
 
 --distinct �ߺ�����
 select*from emp;
 select deptno from emp;
 select distinct deptno from emp;
 
 select job, deptno from emp; --ǥ��Į���� ó���� ���
 select distinct job, deptno from emp;
 
 --���Ῥ����
 select*from emp;
 select job || ename from emp;
 select ename ||'''s job is'|| job from emp;
 
 --where �� ���
 select empno, ename
 from emp; --��� ������ ���
 select empno, ename
 from emp
 where empno=7900; --�������
 
 select*from emp;
 desc emp;
 select empno, ename, sal from emp where sal>2000;

 --empno�� 7700 �̻��� ���, �̸�, ��, �޿��� ���
 select*from emp;
 select empno, ename, job, sal from emp where empno>7700;
 --���� �� ���� ����ϼ�.
 --ename�� king�� ����� ���, �̸�, ���� ���
 select*from emp;
 select empno, ename, job from emp where ename='KING';
 
  select*from emp;
  select ename from emp
  where ename>'MANAGER';
  
  select empno, ename, sal, hiredate
  from emp
  where hiredate >='81/12/10';
  --���� �̻�
 
select*from emp;

--and�� ���
select*from emp
where sal between 2020 and 3000; --2000�̻�~3000����

--or�� ���
select*from emp
where sal <=3000 or sal >=2850;

select sal from emp where sal <=3000 or sal>=2850;

-- emp���̺��� empno�� 7600���� ũ�� sal�� 2000���� ���� ������ ���
select*from emp
where empno >7600 and sal < 2000;

-- emp���̺��� (empno >= 7500�� empno<8000) �̰ų� 
--(sal>=2000�̻��̰� sal<3000����)�� ���� ���
select*from emp 
where empno >= 7500 and empno <8000 or sal>=2000 and sal<=300;

--in ������
select empno, ename, deptno from emp where deptno in(10,20);

--deptno�߿� 10�̰ų� 20�� ������ ���
select*from emp;
select deptno from emp where deptno in(10,20);

--job�� Manager, Saleman�� ������ ���
select*from emp;
select job from emp where job in('MANAGER','SALESMAN');

--job�� president, clerk�̰�, sal>3000�̻��� ������ ��� (in������ Ȱ��)
select*from emp where job in('PRESIDENT','CLERK') and sal >= 300;

--like ������ ***
-- %�� ���� ���� ������ ���� � ���ڶ� �� ���ڸ� �ǹ�
select*from emp where sal like '1%';

--ename���� ù ���ڰ� S�� �����ϴ� ������ ���
select*from emp where ename like 'S%';

--��¥�� ���� 
select*from emp where hiredate like '80%';


-- _�� Ȱ��
select*from emp;
select*from emp where ename like '_L%';
select*from emp where ename like '__L%';

-- null�� Ȱ��
select*from emp
where comm = null; --�̷� ǥ���� �Ұ�

select*from emp
where comm is null; --�´� ǥ�� 

--not null Ȱ��
select*from emp
where comm is not null;

--����ڿ��� �Է��� �޾� ����ϱ� (���� �Է��ϱ�)
select*from emp
where empno=&empno;

select*from &table
where sal>2000; 

select*from emp
where job like '%NA%'; --������ �Է¹޾� LIKE�����ڿ� ����

--20210312 ����

drop table student purge;

create table student
( studno number(4) primary key,
  name   varchar2(30) not null,
  id varchar2(20) not null unique,
  grade number check(grade between 1 and 6),
  jumin char(13) not null,
  birthday  date,
  tel varchar2(15),
  height  number(4),
  weight  number(3),
  deptno1 number(3),
  deptno2 number(3),
  profno  number(4)) tablespace users;

insert into student values (
9411,'James Seo','75true',4,'7510231901813',to_date('1975-10-23','YYYY-MM-DD'),
'055)381-2158',180,72,101,201,1001);

insert into student values (
9412,'Rene Russo','Russo',4,'7502241128467',to_date('1975-02-24','YYYY-MM-DD'),'051)426-1700',172,64,102,null,2001);

insert into student values (
9413,'Sandra Bullock','Bullock',4,'7506152123648',to_date('1975-06-15','YYYY-MM-DD'),'053)266-8947',168,52,103,203,3002);

insert into student values (
9414,'Demi Moore','Moore',4,'7512251063421',
to_date('1975-12-25','YYYY-MM-DD'),'02)6255-9875',177,83,201,null,4001);

insert into student values (
9415,'Danny Glover','Glover',4,'7503031639826',to_date('1975-03-03','YYYY-MM-DD'),'031)740-6388',182,70,202,null,4003);

insert into student values (
9511,'Billy Crystal','Crystal',3,'7601232186327',to_date('1976-01-23','YYYY-MM-DD'),'055)333-6328',164,48,101,null,1002);

insert into student values (
9512,'Nicholas Cage','Cage',3,'7604122298371',to_date('1976-04-12','YYYY-MM-DD'),'051)418-9627',161,42,102,201,2002);

insert into student values (
9513,'Micheal Keaton','Keaton',3,'7609112118379',to_date('1976-09-11','YYYY-MM-DD'),'051)724-9618',177,55,202,null,4003);

insert into student values (
9514,'Bill Murray','Murray',3,'7601202378641',to_date('1976-01-20','YYYY-MM-DD'),'055)296-3784',160,58,301,101,4007);

insert into student values (
9515,'Macaulay Culkin','Culkin',3,'7610122196482',to_date('1976-10-12','YYYY-MM-DD'),'02)312-9838',171,54,201,null,4001);

insert into student values (
9611,'Richard Dreyfus','Dreyfus',2,'7711291186223',to_date('1977-11-29','YYYY-MM-DD'),'02)6788-4861',182,72,101,null,1002);

insert into student values (
9612,'Tim Robbins','Robbins',2,'7704021358674',to_date('1977-04-02','YYYY-MM-DD'),'055)488-2998',171,70,102,null,2001);

insert into student values (
9613,'Wesley Snipes','Snipes',2,'7709131276431',to_date('1977-09-13','YYYY-MM-DD'),'053)736-4981',175,82,201,null,4002);

insert into student values (
9614,'Steve Martin','Martin',2,'7702261196365',to_date('1977-02-26','YYYY-MM-DD'),'02)6175-3945',166,51,201,null,4003);

insert into student values (
9615,'Daniel Day-Lewis','Day-Lewis',2,'7712141254963',to_date('1977-12-14','YYYY-MM-DD'),'051)785-6984',184,62,301,null,4007);

insert into student values (
9711,'Danny Devito','Devito',1,'7808192157498',to_date('1978-08-19','YYYY-MM-DD'),'055)278-3649',162,48,101,null,null);

insert into student values (
9712,'Sean Connery','Connery',1,'7801051776346',to_date('1978-01-05','YYYY-MM-DD'),'02)381-5440',175,63,201,null,null);

insert into student values (
9713,'Christian Slater','Slater',1,'7808091786954',to_date('1978-08-09','YYYY-MM-DD'),'031)345-5677',173,69,201,null,null);

insert into student values (
9714,'Charlie Sheen','Sheen',1,'7803241981987',to_date('1978-03-24','YYYY-MM-DD'),'055)423-9870',179,81,102,null,null);

insert into student values (
9715,'Anthony Hopkins','Hopkins',1,'7802232116784',to_date('1978-02-23','YYYY-MM-DD'),'02)6122-2345',163,51,103,null,null);

commit;

select*from student;

--20210312 ���� 1�� ���� �ش�
select name||'''s ID:'|| id||', WEIGHT is'|| weight||'kg' from student;

--20210312 ���� 2�� ���� �ش�
select ename ||'('||job||'), '||ename|| ''''||job||''''AS "NAME AND JOB"from emp;

--20210312 ���� 3�� ���� �ش�
select ename ||'''s sal is'||'$'|| sal as "Name and Sal" from emp;


-- ORDER BY ���
--asc(��������): 1 2 3 4 a b c d 
--desc(��������): 4 3 2 1 d c b a 
select*from emp;
select ename, sal, hiredate from emp;
select ename, sal, hiredate from emp order by ename; --asc ��������

select ename, sal, hiredate from emp order by ename desc;

--��¥�� �������� ����
select hiredate from emp order by hiredate;

--�޿��� �������� ����
select sal from emp order by sal desc;

--1�� ���İ� 2�� ����, n�� ����
select ename, deptno, sal, hiredate, job
from emp order by deptno asc, sal desc, job asc;


--student ���̺��� �Ʒ� �׸��� ����
-- 1������ 102���� �л����� �̸��� ��ȭ��ȣ, ��ȭ��ȣ���� ���� �κи� ***
--ó���Ͽ� ����ϼ���.
--���
--051)***-1700

select tel from student;
select tel, replace(tel,substr(tel,4,3),'***')from student;
select tel, replace(tel, substr(tel, instr(tel,')')+1,3),'***')from student
where deptno1=102;

select tel,replace(tel, substr(tel, instr(tel,')')+1,3),'***')from student;

select tel,replace(tel, substr(tel, instr(tel,')')+1,  
instr(tel,'-') - instr(tel,')')    ),'***')from student;

select tel,replace(tel, substr(tel, instr(tel,')')+1, 
instr(tel,'-')-instr(tel,')')    ),'***')from student;

-- round(����, ����� �ڸ���) �ݿø��Լ�
select round(987.12345, 4), round(987.12345, -1) from dual;

--update ������ ����
desc dept;
insert into dept values(40, 'OPERATIONS','SEOUL');

select*from dept;

create table dept(
deptno number(4), dname varchar2(20), loc varchar2(20));

insert into dept values(10,'ACCOUNTING','NEW YORK');

insert into dept values(20,'RESEARCH','DALLAS');

insert into dept values(30,'SALES','CHICAGO');

select*from dept;

insert into dept values(40,'OPERATIONS','BOSTON');
insert into dept values(40, 'OPERATIONS','SEOUL');

select*from dept;

commit;

--update ������ ����
desc dept;
insert into dept values(40, 'OPERATIONS','SEOUL');

select*from dept;

--deptno�� 10�� �������� 'dname'�� 'COUNTING'���� ����
update dept
set dname='COUNTING'
where deptno=10;
select*from dept;

update dept
set dname='RESEARCH'
where deptno=30;

select*from dept;

update dept
set dname='SALES',
loc= 'SEOUL'
where deptno=30;

select*from dept;

--40�� �μ��� deptno�� 50���� ����
update dept
set deptno=50
where deptno=40;

select*from dept;

update dept
set deptno=40
where loc='BOSTON';


--job�� MANAGER�� ����� COMM�� 500�� ����
select*from emp;
update emp
set comm=500
where job='MANAGER';

--job�� salesman �̰� comm�� 1000�̸��� ����� comm�� 20% �λ�
select*from emp;
update emp
set comm = comm*1.2
where job='SALESMAN' and COMM < 1000;
rollback;

--3�� 16��
--trunc ����, ����
select trunc(987.456,2),
trunc(987.456,0), trunc(987.456,-1) from dual;

--mod*** ������ �Լ�
--ceil �־��� ������ ���� ����� ����(��)
--floor �־��� ������ ���� ����� ����(����)
select mod(129,10)"mod",
ceil(123.45)"cell",
floor(123.45) "floor",
from dual;

--���з�
select rownum"rownum", ceil

--power(����1, ����2) ����1�� �¼�
select power(2,4) from dual;

--��¥�Լ�
--sysdate***

select sysdate from dual;
select to_char(sysdate,'yyyy-mm-dd')"������ ��¥" from dual;

--month_between(ū ��, ���� ��) �� ��¥ ������ ������ ���
select MONTHS_BETWEEN('14/09/30', '14/08/31') from dual;
select months_between(sysdate, hiredate) from emp;
select months_between(sysdate, hiredate)"months",
round(months_between(sysdate, hiredate),2)"months_round"from emp;

select months_between(sysdate, hiredate)"months",
to_char(round(months_between(sysdate,hiredate),2),'9999.99')"months_round" from emp;

--add_months
select sysdate, add_months(sysdate,1) from dual;

--next_day
select NEXT_DAY(sysdate, '��') from dual;

--last_day
select sysdate,last_day(sysdate) from dual;

--����ȯ(����� ����ȯ(����), ������ ����ȯ(�ڵ�))
--����������ȯ
select 2+'3' from dual;
select 2+to_number('3')from dual;

select 2+'a' from dual;
select 2+ascii('a')from dual;
select ascii ('a')from dual;

select 2+'a' from dual;

--to_char
--yyyy-�⵵
--rrrr-�⵵ 2000�� ���� �⵵
--year-������ �����̸�

select sysdate,
to_char(sysdate,'yyyy')"yyyy",
to_char(sysdate, 'rrrr')"rrrr",
to_char(sysdate, 'yy')"yy",
to_char(sysdate, 'rr')"rr",
to_char(sysdate, 'year')"year"
from dual;

--mm ���ڸ���
--mon ���н��� ����Ŭ���� ���, �����쿡���� month�� ����
--month ���� ���ϴ� ��ü ǥ��
select sysdate,
to_char(sysdate,'mm')"mm",
to_char(sysdate, 'mon')"mon",
to_char(sysdate, 'month')"month"
from dual;

--dd ���� 2�ڸ���
--day ���н� ����Ŭ��, �����쿡�� �ѱ۷�
--ddth �� ��° ������
select sysdate,
to_char(sysdate,'dd')"dd",
to_char(sysdate, 'day')"day",
to_char(sysdate, 'ddth')"ddth"
from dual;

--hh24 24�ð���
--hh 12�ð���
--mi ��
--ss ��
select sysdate, to_char(sysdate, 'rrrr-mm-dd:hh24:mi:ss')from dual;

select*from emp;

--to_char()
--9-9�� ������ŭ �ڸ��� ǥ��
--0- ���ڸ��� 0���� ä��
--$- $ǥ�ø� ����
--.- �Ҽ��� ����ǥ��
--,- õ ���� ǥ��

select to_char (1234, '999999') from dual;
select to_char(1234, '099999') from dual;
select to_char(1234, '$9999') from dual;
select to_char(1234, '9999.99')from dual;
select to_char(12341234, '999,999,999')from dual;

--����
select*from emp;
select ename,sal,comm, to_char((sal*12)+comm, '$999,999')"salary" from emp;

--���빮��
--professor ���̺��� ��ȸ�Ͽ� 201�� �а��� �ٹ��ϴ� �������� �̸��� 
--�޿�,���ʽ�,������ �Ʒ��� ���� ����ϼ���. ��, ������ (pay*12)+bonus
--�� ����մϴ�.

select*from professor;
select name, pay, bonus, to_char((pay*12)+bonus, '$999,999')"PAY"
from professor
where deptno=201;

--���빮��
--emp ���̺��� ��ȸ�Ͽ� comm ���� ������ �ִ� ������� empno, ename, hiredate,
--�ѿ���, 15%�λ� �� ������ �Ʒ� ȭ�� ó�� ����ϼ���. ��, �ѿ����� 
--(sal*12)+comm���� ����ϰ� �Ʒ� ȭ�鿡���� SAL�� ��µǾ�����
--15% �λ��� ���� �ѿ����� 15% �λ� ���Դϴ�.
--(HIREDATE Į���� ��¥ ���İ� SAL Į��, 15% UP Į���� $ ǥ�ÿ�, ��ȣ ������ 
--�ϼ���

select*from emp;
select empno, ename, to_char(hiredate, 'yyyy-dd-mm')"HIREDATE"
, sal, to_char(((sal*12)+comm)*1.15, '$999,999')"15% �λ�"
from emp
where comm is not null;

--to_number
select to_number('5')from dual;
select to_number('a')from dual; --������
select ascii('a') from dual; --���� a�� �ƽ�Ű�ڵ尪���� ����

--to_date
select to_date('2014/05/05') from dual;

--nvl(null value�� ����) *****
--nvl(col,0) col�� null�̸� 0���� ���
select*from emp;

select ename, comm, nvl(comm,0) from emp;

--nvl ����
-- professor ���̺��� 201�� �а� �������� �̸��� �޿�, bonus,
-- �� ������ �Ʒ��� ���� ����ϼ���. ��, �� ������(pay*12+bonus)��
-- ����ϰ� bonus�� ���� ������ 0���� ����ϼ���.

select*from professor;
select name, bonus, pay, to_char(pay*12+nvl(bonus,0),'$999,999')"�� ����"
from professor
where deptno = 201;

--nvl2(col1, col2, col3) col1�� null�� �ƴϸ� col2, col1�� null�̸� col3
--nvl2 ����
--emp ���̺��� deptno�� 30���� ������� empno, ename, sal, comm ����
--����ϵ� ���� comm ���� null�� �ƴϸ� sal+comm ���� ����ϰ� comm ����
--null �̸� sal*0�� ���� ����ϼ���.
select empno, ename, sal, comm,
nvl2(comm,sal+comm,sal+0)
from emp;

--nvl2 ����
--�Ʒ� ȭ��� ���� emp ���̺��� deptno�� 30���� 
--������� empno, ename, comm ��ȸ�Ͽ�
--comm ���� ���� ��� 'Exist'�� ����ϰ� comm ���� null�� ��� 'NULL'�� ����ϼ���.

select*from emp;
select empno, ename, comm, nvl2(comm, 'Exist', 'NULL')
from emp
where deptno=30;

--decode(a,b,'1',null) a�� b�̸� 1�� ���, null�������� --if�Լ�********
select deptno, name, decode(deptno, 101,'Computer Engineering')"dname"
from professor;

select*from professor;

--decode(a,b,'1','2')
select deptno, name, decode(deptno,101,'Computer Engineering','etc')"dname"
from professor;

decode(a,b,1,c,2,d,3,4)
select deptno, name, decode(deptno, 101, 'Computer Engineering',
101, 'Computer Engineering1',
102, 'Computer Engineering2',
103, 'Computer Engineering3',
201, 'Computer Engineering1',
202, 'Computer Engineering2',
203, 'Computer Engineering3',
301, 'Bigdata Science1',
301, 'Bigdata Science2'
)"dname"
from professor;
select*from professor;

--decode(a,b,decode(c,d,'1',null))a�� b�� ��쿡�� c�� d�̸� 1 ��� �ƴ� ���
--null.��ødecode
select deptno,name,decode(deptno,101,decode(name,'Audie Murphy','Best11'))"etc"
from professor;

select deptno,name,decode(deptno,101,decode(name,'Audie Murphy',
'Best11', 'N/A'))"etc"
from professor;

--case Ȱ��
select*from student;
select name,tel
from student;

--tel���� ������ȣ�� ����ϴ� Į���� tel2�� ��� ex) 051
select name, tel,
substr(tel,1,instr(tel,')')-1)tel2
from student;

--�� ������ case�� �����ϱ�
select name,tel,
case(substr(tel,1,instr(tel,')')-1)) when '02'then 'seoul' 
when '031' then 'gyeonggi'
when '051' then 'pusan'
when '031' then 'ulsan'
when '055' then 'gyeongnam'
else 'etc'
end tel2
from student;

--case Ȱ�빮��
student ���̺��� jumin Į���� �����Ͽ� �л����� �̸��� �¾ ��, 
�׸��� �б⸦ ����ϼ���. �¾ ���� 01-03���� 1/4, 04-06���� 2/4,
07-09���� 3/4, 10-12���� 4/4�� ����ϼ���.

select*from student;
select name, substr(jumin,3,2)"month",
case when substr(jumin,3,2) 
between '01' and '03' then '1/4'
when substr(jumin,3,2) between '04' and '06' then '2/4'
when substr(jumin,3,2) between '07' and '09' then '3/4'
when substr(jumin,3,2) between '10' and '12' then '4/4'
end "qua"
from student;

select*from dept;

select*from emp;

select*from dept;
update dept
set loc='BOSTON'
where deptno=40 and loc='SEOUL';

alter table dept add seq_no number(5);
select*from dept;

update dept set seq_no=rownum;
delete dept
where seq_no=5;
select*from dept;
alter table dept drop column seq_no;

commit;

--2021/03/17
--decode grade�� �������� name, grade, �г� ���
--1-1�г�
--2-2�г�

select*from student;
select name, grade, decode(grade,1,'1�г�',null)"1�г�",
decode(grade,2,'2�г�',null)"2�г�",
decode(grade,3,'3�г�',null)"3�г�"
,decode(grade,'4�г�',null)"4�г�"
from student;

--���輳���� ���� ����
create table one(
no number, name varchar2(10));

create table two(
num number, addr varchar2(30));

insert into two values(10,'SEOUL');
select*from two;
delete from two;

--�������� two ���̺� �߰� ���������� �̸��� fk_num
alter table two add constraint fk_num foreign key(num) 
references one(no); --���� �Ҵ�

--one ���̺� primary key �߰�
alter table one add constraint pk_one_no primary key(no);

--one ���̺� �⺻Ű�� �߰� �� �������� two���̺� �߰� ���������� �̸���
--fk_num
alter table two add constraint fk_num foreign key(num) references one(no);

--two���̺� ������ �߰�
insert into two values(10,'seoul'); --���Ἲ ���� ��������,
-- Integrity constraint (HR.FK_NUM) violated

select*from one;
insert into one values(10,'blue');
insert into one values(20,'blue');

insert into two values(10,'seoul'); --one ���̺� 10�� ���ڵ� �߰��� ���� ������
insert into two values(20,'pusan');

--�������� ��ȸ
select*from all_constraints
where table_name='TWO';

--�������� ����
alter table two drop constraint fk_num;

insert into two values(30,'seoul');

insert into two values(10,'seoul'); -- one ���̺� 10�� ���ڵ� �߰��� ���� ������
insert into two values(20,'pusan');

select*from one;
delete from one where no=10  cascade;

alter table one drop column no;

alter table one drop constraint pk_one_no cascade;

select*from one;

select*from two;
delete from two;

CREATE TABLE EMP (
 EMPNO               NUMBER(4) NOT NULL,
 ENAME               VARCHAR2(10),
 JOB                 VARCHAR2(9),
 MGR                 NUMBER(4) ,
 HIREDATE            DATE,
 SAL                 NUMBER(7,2),
 COMM                NUMBER(7,2),
 DEPTNO              NUMBER(2) );



INSERT INTO EMP VALUES (7839,'KING','PRESIDENT',NULL,'81-11-17',5000,NULL,10);
INSERT INTO EMP VALUES (7698,'BLAKE','MANAGER',7839,'81-05-01',2850,NULL,30);
INSERT INTO EMP VALUES (7782,'CLARK','MANAGER',7839,'81-05-09',2450,NULL,10);
INSERT INTO EMP VALUES (7566,'JONES','MANAGER',7839,'81-04-01',2975,NULL,20);
INSERT INTO EMP VALUES (7654,'MARTIN','SALESMAN',7698,'81-09-10',1250,1400,30);
INSERT INTO EMP VALUES (7499,'ALLEN','SALESMAN',7698,'81-02-11',1600,300,30);
INSERT INTO EMP VALUES (7844,'TURNER','SALESMAN',7698,'81-08-21',1500,0,30);
INSERT INTO EMP VALUES (7900,'JAMES','CLERK',7698,'81-12-11',950,NULL,30);
INSERT INTO EMP VALUES (7521,'WARD','SALESMAN',7698,'81-02-23',1250,500,30);
INSERT INTO EMP VALUES (7902,'FORD','ANALYST',7566,'81-12-11',3000,NULL,20);
INSERT INTO EMP VALUES (7369,'SMITH','CLERK',7902,'80-12-09',800,NULL,20);
INSERT INTO EMP VALUES (7788,'SCOTT','ANALYST',7566,'82-12-22',3000,NULL,20);
INSERT INTO EMP VALUES (7876,'ADAMS','CLERK',7788,'83-01-15',1100,NULL,20);
INSERT INTO EMP VALUES (7934,'MILLER','CLERK',7782,'82-01-11',1300,NULL,10);


select*from emp;

--emp ���̺� primary Ű �߰��ϱ�
alter table emp add constraint pk_emp_empno primary key(empno);

--emp ���̺� �������� �߰��ϱ�
alter table emp add constraint fk_empno foreign key(empno) 
references emp(empno);


--2021/03/17 �ǽ�
create table rental(HAKBUN number(20),NAME varchar2(20),CITY varchar(20)
,LOCKER number);

select*from rental;
select*from lockerroom;

select*from all_constraints where table_name='rental';
select*from all_constraints where table_name='lockerroom';

--lockerroom ���̺� �⺻Ű�� �߰��� �������� rental���̺� �߰� 
--���������� �̸��� fk_locker
alter table rental add constraint fk_locker foreign key(locker) 
references lockerroom(no);

insert into rental values(1111,'������','����',1);
insert into rental values(2222,'������','�뱸',2);
insert into rental values(3333,'������','�λ�',3);

create table lockerroom(no number primary key, name varchar2(50));
insert into lockerroom values(1,'1�� �繰��');
insert into lockerroom values(2, '2�� �繰��');
insert into lockerroom values(3, '3�� �繰��');

--�׷��Լ�
--count() ����
select*from emp;
select count(comm), count(mgr), count(*) from emp;

--sum()
select count(comm), sum(comm)"sum" from emp;

--avg() ��� -- ���� ���� null�� ������ ���, ��ü���
select count(comm), sum(comm), avg(comm) from emp;
select count(comm)"comm �ο�", count(*)"���ο�",sum(comm),avg(comm),
sum(comm)/count(*)"��ü���" from emp;
--avg�� ���ο��� ���� ����� �ƴ϶� comm�� �ο��� ���� ������� �ǹ��Ѵ�.
--��ü ����� ���ϰ� ������ sum/count(*)�� ����ϸ� �ȴ�.
-- Ȥ�� avg(nvl(comm,0))�� �̿��� ���� �ִ�.


--�Ϲ��Լ��� �ߺ��ؼ� ����ϱ� ��ü���
select count(comm)"comm�ο�",count(*)"���ο�", sum(comm), avg(comm), 
round(avg(nvl(comm,0)),2)"��ü���" from emp;

--max min
select max(sal), min(sal)
from emp;

select max(hiredate)"max",
min(hiredate)"min"
from emp;

--stddev ǥ������, variance �л�
select round(stddev(sal),2),
round(variance(sal),2)
from emp;

--�׷��Լ� ���� ���� group by�� ���� �̷������ ��°���
-- �Ʒ��� ���� ��� ���� �����̴�.
--from
--where
--group by
--having
--select
--order by

--group by
select*from emp;
select deptno, avg(sal)"avg"
from emp
group by deptno;

select avg(sal)"avg"
from emp
group by deptno;

--��ø�׷�
select deptno, job, avg(sal)"avgsal"
from emp
group by deptno, job
order by deptno, job asc;

--���� select ������ ���� Į���� groupȭ ���Ѿ� �Ѵ�. 
select job,deptno,avg(sal)"avgsal"
from emp
group by deptno, job
order by 1,2;

--having �׷��� ����
select*from emp;
select deptno, avg(nvl(sal,0))
from emp
where avg(nvl(sal,0))>2000; --����

select deptno,avg(nvl(sal,0))
from emp
group by deptno
having avg(nvl(sal,0))>=2000;

select*from professor;

--deptno�� ���� count �� pay ���� ���
select*from professor;
select deptno,count(*),sum(pay)
from professor
group by deptno;

--emp���̺��� 2005�� �μ��� ����� ���
select '2005��',deptno"�μ���ȣ", count(*)"�����"
from emp
group by deptno;

--�μ����� �׷��Ͽ� �μ���ȣ, �ο���, �޿��� ���, �޿��� ���� ��ȸ
select*from emp;
select deptno, count(*), avg(sal), sum(sal)
from emp
group by deptno;

--�������� �׷��Ͽ� ����, �ο���,��� �޿���, �ְ� �޿���, ���� �޿��� �� �հ� ��ȸ
select*from emp;
select job, count(*)"�ο���", round(avg(sal),2)"���",
max(sal)"�ְ��ӱݾ�", min(sal)"�����ӱݾ�", sum(sal)"�޿��հ�"
from emp
group by job;

--having ����
--��� ���� �ټ� ���� �Ѵ� �μ��� ��� ���� ��ȸ
select*from emp;
select deptno, count(*)"�����"
from emp
group by deptno
having count(*)>5;

--��ü ������ 5000�� �ʰ��ϴ� JOB�� ���ؼ� JOB�� ������ �հ踦 ��ȸ�ϴ� ���̴�.
--�� �Ǹſ�(SALES)�� �����ϰ� �� �޿� �հ�� �������� ����

select*from emp;
select job,sum(sal)"�޿��հ�"             
from emp
where job !='SALESMAN'
group by job
having sum(sal)>5000
order by sum(sal) desc;

create table ����(���� varchar2(20));
insert into ���� values('�����ڵ�');
insert into ���� values('������');
insert into ���� values('ȭ���');
select*from ����;

create table �����(����� varchar2(20));
insert into ����� values('ID');
insert into ����� values('�̸�');
insert into ����� values('�����ڵ�');
insert into ����� values('����');
insert into ����� values('����');
select*from �����;


alter table ���� add constraint fk_country primary key(����);
alter table ����� add constraint fk_user primary key(�����);


select*from all_constraints
where table_name='����';

select*from all_constraints
where table_name='�����';

---����Ŭ ���� ����
--1�� ����
select*from emp;
select empno, ename, sal
from emp
where deptno = 10;

--2�� ����
select*from emp;
select ename, hiredate, deptno
from emp
where deptno = 7369;

--3�� ����
select*from emp where ename='ALLEN';

--4�� ����
select*from emp;
select ename, deptno, sal 
from emp
where hiredate='83/01/12';

--5�� ����
select*from emp;
select*from emp where job != 'MANAGER';

--6�� ����
select*from emp where hiredate > '81/04/02';

--7�� ����
select ename,sal,deptno
from emp
where sal > 800;

--8������
select*from emp where deptno >= 20;

--9������
select*from emp where ename > 'K';

--10������
select*from emp where hiredate < '81/12/09';

--11�� ����
select empno,ename
from emp
where empno <= 7698;

--12�� ����
select ename, sal, deptno
from emp
where hiredate > '81/04/02' and hiredate < '82/12/09';

--13�� ����
select ename, job, sal
from emp
where sal > 1600 and sal < 8000;

--14�� ����
select*from emp where empno >= 7654 and empno <= 7782;

--15�� ����
select*from emp where ename >= 'B' and ename <='J';

--16�� ����
select*from emp where hiredate > '82/01/01' or hiredate < '80/12/31';

--17�� ����
select*from emp where job='MANAGER' or job='SALESMAN';

--18�� ����
select ename, empno, deptno
from emp
where deptno != 20 and deptno != 30;

--19�� ����
select*from emp where ename like 'S_%';

--20�� ����
select*from emp where hiredate >='81/01/01' and hiredate <= '81/12/31';

--21�� ����
select*from emp where ename like('%S%');

--22�� ����
select*from emp where ename like 'S___T';

--23�� ����
select*from emp where ename like '_A%';

--24�� ����
select*from emp where COMM is null;

--25�� ����
select*from emp where COMM is NOT null;

--26�� ����
select ename,deptno, sal
from emp
where deptno=30 and sal>1500;

--27�� ����
select ename, empno, deptno
from emp
where ename like 'K_%' or deptno=30;

--28�� ����
select*from emp where sal > 1500 and deptno=30 and job='MANAGER';

commit;

--3/17 ����
--sum() �հ�
select count(comm), sum(comm)"sum" from emp;

--avg()���
select count(comm),sum(comm),avg(comm) from emp;
select count(comm)"comm�ο�",count(*)"���ο�", sum(comm),avg(comm) from emp;

--20210318 ����
-- join
-- ���̺� �����
create table tbl1(id int, name varchar2(10));

insert into tbl1 values(1,'aaa');
insert into tbl1 values(2,'bbb');
insert into tbl1 values(3,'ccc');
insert into tbl1 values(4,'ddd');
insert into tbl1 values(5,'eee');
select*from tbl1;

create table tbl2(id int, car varchar2(10));
insert into tbl2 values(2,'AVANTE');
insert into tbl2 values(3,'SONATA');
insert into tbl2 values(2,'MINI');
insert into tbl2 values(6,'PONY');
select*from tbl2;

--inner join ������, ���� ���̺��� �������� ������ ���̺� ��ġ�ϴ� ������ ��ȸ
select a.id, name, car
from tbl1 a inner join tbl2 b 
on a.id=b.id;  --ansl style

select a.id,a.name,b.car
from tbl1 a, tbl2 b
where a.id=b.id; --oracle style

--left outer join
--���� ��Ҵ� �� �����ְ� �����ʲ��� ���ʰ� ��ġ�ϴ� ���� ���
select a.id, a.name, b.car
from tbl1 a left outer join tbl2 b
on a.id=b.id;  ---ansl join

select a.id,a name, b.car
from tbl2, b left outer join tbl1 a
on a.id=b.id;

select  b.car,a.id,a.name
from tbl2 b left outer join tbl1 a
on a.id=b.id;

select a.id,a.name,b.car
from tbl1 a,tbl2 b
where a.id=b.id(+); --oracle join

--right outer join
--������ ��Ҵ� �� �����ְ� ���ʲ��� �����ʰ� ��ġ�ϴ� ���� ���
 select a.id,a.name,b.car
 from tbl1 a right outer join tbl2 b
 on a.id=b.id ; --ansl join

select a.id,a.name,b.car
from tbl1 a, tbl2 b
where a.id(+)=b.id;  --oracle join

create table stu(sno number(20), sname  varchar2(20));
insert into stu values(100, 'torm');
insert into stu values(101, 'smith');
insert into stu values(102, 'brown');
insert into stu values(103, 'bell');
insert into stu values(104, 'clark');
insert into stu values(105, 'evans');
select*from stu;

create table lecture(sno number(20), lec varchar2(20));
insert into lecture values(100, 'com science');
insert into lecture values(102, 'programming');
insert into lecture values(107, 'graphic');
insert into lecture values(102, 'network');
select*from lecture;


select a.sno, a.sname, b.lec
from stu a inner  join lecture b
on a.sno=b.sno;  ---ansl inner  join

select a.sno,a.sname,b.lec
from stu a, lecture b
where a.sno=b.sno; --oracle inner join

select a.sno, a.sname, b.lec
from stu a left outer join  lecture b
on a.sno=b.sno;  --ansl left outer join

select a.sno,a.sname,b.lec
from stu a, lecture b
where a.sno=b.sno(+); --oracle left outer join

select a.sno, a.sname, b.lec
from stu a right outer join  lecture b
on a.sno=b.sno;  --ansl right outer join

select a.sno,a.sname,b.lec
from stu a, lecture b
where a.sno(+)=b.sno; --oracle rightt outer join


--left join�� �̿��ؼ� inner join�� ǥ��
select  a.id,a.name,b.car
from tbl1 a left outer join tbl2 b
on a.id=b.id
where b.id is not null;


--right join�� �̿��ؼ� ineer join�� ǥ��
select  a.id,a.name,b.car
from tbl1 a right outer join tbl2 b
on a.id=b.id
where a.id is not null; 

--�̼ǿϷ�

--join
create table cat_a(no number, name varchar2(1));

insert into cat_a values(1,'A');
insert into cat_a values(2,'B');

create table cat_b(no number, name varchar2(1));
insert into cat_b values(1,'C');
insert into cat_b values(2,'D');

create table cat_c(no number, name varchar2(1));
insert into cat_c values(1,'E');
insert into cat_c values(2,'F');
commit;

--īƼ�ǰ� -����� �� ��� ��� ������ ������ ����
select a.name, b.name, c.name
from cat_a a, cat_b b,  car_c c;

--a b c�� no�� � ����
select a.name, b.name, c.name
from cat_a a, cat_b b, cat_c c
where a.no=b.no
and b.no=c.no;

--3���� ���̺� ��ȸ���� ������������ 2�� ���̺� ������ ���� īƼ�ǰ�
select a.name, b.name, c.name
from cat_a a, cat_b b, cat_c c
where a.no=b.no;

--�����(equl join)
--���������� ������ =�� ����� ����
--emp ���̺� Ȱ��
select empno, ename, dname
from emp e, dept d
where e.deptno  = d.deptno;

select empno, ename, dname
from emp e join dept d
on e.deptno=d.deptno;

--���� ��뿡
--�л� ���̺�(student) �� ���� ���̺�(professor)��
--join �Ͽ� �л��� �̸��� ����������ȣ, �������� �̸��� ����ϼ���.
select*from student;
select*from professor;

select  s.name, p.profno, p.name
from student s, professor p
where  s.profno= p.profno;

--student  ���̺��� 1������ 101�� �л��� �̸��� �� �л����� ���������� ���
select  s.name, p.name,s.deptno1
from student s, professor p
where s.profno=p.profno
and s.deptno1=101;

create table customer
(gno  number(8) ,
 gname varchar2(30) ,
 jumin char(13) ,
 point number) ;

insert into customer values (20010001,'James Seo','7510231369824',980000);
insert into customer values (20010002,'Mel Gibson','7502241128467',73000);
insert into customer values (20010003,'Bruce Willis','7506152123648',320000);
insert into customer values (20010004,'Bill Pullman','7512251063421',65000);
insert into customer values (20010005,'Liam Neeson','7503031639826',180000);
insert into customer values (20010006,'Samuel Jackson','7601232186327',153000);
insert into customer values (20010007,'Ahnjihye','7604212298371',273000);
insert into customer values (20010008,'Jim Carrey','7609112118379',315000);
insert into customer values (20010009,'Morgan Freeman','7601202378641',542000);
insert into customer values (20010010,'Arnold Scharz','7610122196482',265000);
insert into customer values (20010011,'Brad Pitt','7711291186223',110000);
insert into customer values (20010012,'Michael Douglas','7704021358674',99000);
insert into customer values (20010013,'Robin Williams','7709131276431',470000);
insert into customer values (20010014,'Tom Hanks','7702261196365',298000);
insert into customer values (20010015,'Angela Bassett','7712141254963',420000);
insert into customer values (20010016,'Jessica Lange','7808192157498',598000);
insert into customer values (20010017,'Winona Ryder','7801051776346',625000);
insert into customer values (20010018,'Michelle Pfeiffer','7808091786954',670000);
insert into customer values (20010019,'Whoopi Goldberg','7803242114563',770000);
insert into customer values (20010020,'Emma Thompson','7802232116784',730000);
commit ;

select*from customer;

create table gift
( gno  number ,
  gname varchar2(30) ,
  g_start  number ,
  g_end  number );

insert into gift values(1,'Tuna Set',1,100000);
insert into gift values(2,'Shampoo Set',100001,200000);
insert into gift values(3,'Car wash Set',200001,300000);
insert into gift values(4,'Kitchen Supplies Set',300001,400000);
insert into gift values(5,'Mountain bike',400001,500000);
insert into gift values(6,'LCD Monitor',500001,600000);
insert into gift values(7,'Notebook',600001,700000);
insert into gift values(8,'Wall-Mountable TV',700001,800000);
insert into gift values(9,'Drum Washing Machine',800001,900000);
insert into gift values(10,'Refrigerator',900001,1000000);
commit ;

select*from gift;

--�� ����(Non-Equl join)
--Customer ���̺�� gift ���̺��� Join�Ͽ� ������ ���ϸ��� ����Ʈ�� ��ȸ�� ��
--�ش� ���ϸ��� ������ ���� �� �ִ� ��ǰ�� ��ȸ�Ͽ� ���� �̸��� ���� �� �ִ� ��ǰ ���� 
--�Ʒ��� ���� ����ϼ���.

--ORACLE JOINT
select*from customer;
select*from gift;
select c.gname "CUST_NAME",to_char(c.point,'999,999')"POINT", g.gname"GIFT_SET"
from customer c, gift g
where c.point between g.g_start  and g.g_end;

--JOIN ON ����ؼ� ANSI JOIN
select c.gname"CUST_NAME",to_char(c.point,'999,999')"POINT",g.gname"GIFT_SET"
from customer c join gift g
on c.point between g.g_start and g.g_end;

--�� ������ �� �����ڷ� ����
select c.gname"CUST_NAME",to_char(c.point,'999,999')"POINT",g.gname"GIFT_SET"
from customer c join gift g
on c.point  >= g.g_start 
and c.point <=  g.g_end;


create table score
(studno  number ,
 total number);

insert into score values (9411,97);
insert into score values (9412,78);
insert into score values (9413,83);
insert into score values (9414,62);
insert into score values (9415,88);
insert into score values (9511,92);
insert into score values (9512,87);
insert into score values (9513,81);
insert into score values (9514,79);
insert into score values (9515,95);
insert into score values (9611,89);
insert into score values (9612,77);
insert into score values (9613,86);
insert into score values (9614,82);
insert into score values (9615,87);
insert into score values (9711,91);
insert into score values (9712,88);
insert into score values (9713,82);
insert into score values (9714,83);
insert into score values (9715,84);

commit ;
select*from score;

create table hakjum
(grade char(3) ,
 min_point  number ,
 max_point  number );

insert into hakjum values ('A+',96,100);
insert into hakjum values ('A0',90,95);
insert into hakjum values ('B+',86,89);
insert into hakjum values ('B0',80,85);
insert into hakjum values ('C+',76,79);
insert into hakjum values ('C0',70,75);
insert into hakjum values ('D',0,69);

commit;
select*from hakjum;

--student ���̺��� score, hakjum ���̺��� ��ȸ
--�л��̸� ���� ������ ��ȸ
select*from student;
select*from hakjum;
select*from score;

--student ���̺�� score, hakjum ���̺��� ��ȸ
--�л��̸� ���� ������ ��ȸ
--���ȭ��
--name  ����  ����
--Tom  62  D

select s.name, c.total, h.grade
from student s, score c, hakjum h
where s.studno = c.studno
and c.total >= h.min_point
and c.total <= h.max_point; --ORACLE JOIN

select s.name, c.total, h.grade
from student s join score c
on  s.studno = c.studno
join hakjum h
on c.total >= h.min_point
and c.total <= h.max_point; --ANSI JOIN


--SELF JOIN
select*from emp;
--CLARK�� ��� KING �Դϴ�.
select e1.ename, e2.ename
from emp e1, emp e2
where e1.mgr=e2.empno;

select  e1.ename"����̸�", e2.ename"�Ŵ����̸�"
from emp e1, emp e2
where e1.mgr=e2.empno;-- ORACLE JOIN

select e1.ename, e2.ename
from emp e1 join emp e2
on e1.mgr=e2.empno;  --ANSI JOIN

--SUBQUERY ���� �ӿ� ����
--����ϴ� ��
select*from emp;
select ename
from emp
where sal > (select sal from emp where ename='ALLEN');

select sal
from emp
where ename='ALLEN';

update emp
set sal='4000'
where ename='ALLEN';

--emp WARD �� COMM���� ���� ����� �̸��� Ŀ�̼� ���
select ename, comm
from emp
where comm < (select comm from emp where ename='WARD');

--���������� ������
--1. SUBQUERY �κ��� WHERE ���� ������ �����ʿ� ��ġ�ϸ� �ݵ�� (  )�� ����� �Ѵ�.
--2. Ư���� ��츦 �����ϰ� SUBQUERY ������ ORDER BY�� ����� �� ����.
--3. ������ SUBQUERY, ������ SUBQUERY�� �����ڸ� �� �����Ͽ��� �Ѵ�.

--������ ��������
-- =, <>, > , >=, <, >=

create table department
( deptno number(3) primary key ,
  dname varchar2(50) not null,
  part number(3),
  build  varchar2(30))tablespace users;

insert into department 
values (101,'Computer Engineering',100,'Information Bldg');

insert into department
values (102,'Multimedia Engineering',100,'Multimedia Bldg');

insert into department
values (103,'Software Engineering',100,'Software Bldg');

insert into department
values (201,'Electronic Engineering',200,'Electronic Control Bldg');

insert into department
values (202,'Mechanical Engineering',200,'Machining Experiment Bldg');

insert into department
values (203,'Chemical Engineering',200,'Chemical Experiment Bldg');

insert into department
values (301,'Library and Information science',300,'College of Liberal Arts');

insert into department
values (100,'Department of Computer Information',10,null);

insert into department
values (200,'Department of Mechatronics',10,null);

insert into department
values (300,'Department of Humanities and Society',20,null);

insert into department
values (10,'College of Engineering',null,null);

insert into department
values (20,'College of Liberal Arts',null,null);

commit;

--student ���̺�� department ���̺��� ����Ͽ� 'Anthony Hopkins' �л���
-- 1����(deptno1)�� ������ �л����� �̸��� 1���� �̸��� ���

select*from student;
select*from department;

select s.name,d.dname
from student s, department d
where s.deptno1 =  d.deptno
and s. deptno1= 
(select deptno1 from student where name='Anthony Hopkins'); 

--student ���̺��� 1����(deptno1)�� 201���� �а� ��� �����Ժ��� �����԰� ����
--�л��� �̸��� ������ ���
--1. student ���̺��� 1����(deptno1)�� 201�� ���� Ȯ��

select name, weight
from student
where weight > (select avg(weight) from student where deptno1 = 201);

-------------------------------
--�����̸�: sql318_1_40  ���伷.txt
--ī��������
--------------------------------
select*from emp;
--29�� ����
select  empno from emp where deptno = 30;

--30�� ����
select *from  emp  order by  sal desc;

--31�� ����
select*from emp order by deptno asc, sal desc;

--32�� ����
select*from emp order by deptno desc, ename asc, sal desc;

--33�� ����
select ename, sal, comm, sal+comm"�Ѿ�" from emp order by sal desc;

--34�� ����
select ename, sal, sal*1.13"bonus", deptno
from emp;

--35�� ����
select ename, deptno, sal, sal*11+sal*1.5"����"
from emp where deptno=30;

--36�� ����
select ename, sal, round(sal/12/5,1)" �ð��� �ӱ�" from emp where deptno=20; 

--37�� ����
select ename, sal, round(sal*0.15,2)" ȸ��" from emp where sal >= 1500 and sal <= 3000;

--38�� ����
select ename, sal, sal*0.15"������" from emp where sal >= 2000;

--39�� ���� --������ �𸣰���
select*from emp;
select  deptno, ename, hiredate, sysdate, date(between  hiredate and sysdate)"�ٹ��ϼ�",
date((between  hiredate and sysdate)/365)"�ٹ����", date((between  hiredate and sysdate)/365*12)"�ٹ�����",
date((between  hiredate and sysdate)/7)"�ٹ��ּ�"
from emp;

--40�� ����
select ename, sal, sal-sal*0.1"�Ǽ��ɾ�"   from emp order by sal desc;

------------------------------
commit;