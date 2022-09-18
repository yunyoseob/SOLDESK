show user;
select*from countries;

create table sawon(
name varchar2(20),
grade varchar2(10),
job number
);

select*from sawon;--사원테이블의 내용을 출력
desc sawon;

--data 추가
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

--sawon2 테이블 만들기
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
--sano 속성에 primary key 제약으로 오류남
insert into sawon values(40, null, 'incheon');
--saname 속성에 not null 제약으로 오류남

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
--테이블의 일부 데이터 출력
select empno, job, hiredate from emp;
select empno, job, to_char(hiredate,'yyyy-nn-dd') from emp;
--다른 툴에서 날짜 입력시 오류가 날때 해결

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
--name 필드 출력
select name from professor;
--필드 이외 내용출력
select name, 'good morng'"Good Morning"from professor;
--별칭(Alias)*****
select*from professor;
select profno as "교수번호", name as "교수이름",
id as "교수아이디",
position as "교수지위",
pay as "급여",
email as "이메일" from professor;
select*from professor;

--보너스 인상
select profno, name, id, position, pay*1.1 from professor;
select profno, name, id, position, pay*1.1 as "인상금액10%" from professor;

select profno 교수번호,
name as "교수이름" from professor;
 
 --distinct 중복제거
 select*from emp;
 select deptno from emp;
 select distinct deptno from emp;
 
 select job, deptno from emp; --표현칼럼의 처음에 사용
 select distinct job, deptno from emp;
 
 --연결연산자
 select*from emp;
 select job || ename from emp;
 select ename ||'''s job is'|| job from emp;
 
 --where 절 사용
 select empno, ename
 from emp; --모든 데이터 출력
 select empno, ename
 from emp
 where empno=7900; --출력조건
 
 select*from emp;
 desc emp;
 select empno, ename, sal from emp where sal>2000;

 --empno가 7700 이상인 사번, 이름, 잡, 급여를 출력
 select*from emp;
 select empno, ename, job, sal from emp where empno>7700;
 --작은 값 먼저 기록하셍.
 --ename이 king인 사람을 사번, 이름, 잡을 출력
 select*from emp;
 select empno, ename, job from emp where ename='KING';
 
  select*from emp;
  select ename from emp
  where ename>'MANAGER';
  
  select empno, ename, sal, hiredate
  from emp
  where hiredate >='81/12/10';
  --오전 이상
 
select*from emp;

--and일 경우
select*from emp
where sal between 2020 and 3000; --2000이상~3000이하

--or일 경우
select*from emp
where sal <=3000 or sal >=2850;

select sal from emp where sal <=3000 or sal>=2850;

-- emp테이블에서 empno가 7600보다 크고 sal가 2000보다 작은 데이터 출력
select*from emp
where empno >7600 and sal < 2000;

-- emp테이블에서 (empno >= 7500고 empno<8000) 이거나 
--(sal>=2000이상이고 sal<3000이하)인 값을 출력
select*from emp 
where empno >= 7500 and empno <8000 or sal>=2000 and sal<=300;

--in 연산자
select empno, ename, deptno from emp where deptno in(10,20);

--deptno중에 10이거나 20인 데이터 출력
select*from emp;
select deptno from emp where deptno in(10,20);

--job이 Manager, Saleman인 데이터 출력
select*from emp;
select job from emp where job in('MANAGER','SALESMAN');

--job이 president, clerk이고, sal>3000이상인 데이터 출력 (in연산자 활용)
select*from emp where job in('PRESIDENT','CLERK') and sal >= 300;

--like 연산자 ***
-- %는 글자 수에 제한이 없고 어떤 글자라도 한 글자를 의미
select*from emp where sal like '1%';

--ename에서 첫 글자가 S로 시작하는 데이터 출력
select*from emp where ename like 'S%';

--날짜에 적용 
select*from emp where hiredate like '80%';


-- _의 활용
select*from emp;
select*from emp where ename like '_L%';
select*from emp where ename like '__L%';

-- null의 활용
select*from emp
where comm = null; --이런 표현은 불가

select*from emp
where comm is null; --맞는 표현 

--not null 활용
select*from emp
where comm is not null;

--사용자에게 입력을 받아 출력하기 (값을 입력하기)
select*from emp
where empno=&empno;

select*from &table
where sal>2000; 

select*from emp
where job like '%NA%'; --변수를 입력받아 LIKE연산자에 적용

--20210312 과제

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

--20210312 과제 1번 문제 해답
select name||'''s ID:'|| id||', WEIGHT is'|| weight||'kg' from student;

--20210312 과제 2번 문제 해답
select ename ||'('||job||'), '||ename|| ''''||job||''''AS "NAME AND JOB"from emp;

--20210312 과제 3번 문제 해답
select ename ||'''s sal is'||'$'|| sal as "Name and Sal" from emp;


-- ORDER BY 행렬
--asc(오름차순): 1 2 3 4 a b c d 
--desc(내림차순): 4 3 2 1 d c b a 
select*from emp;
select ename, sal, hiredate from emp;
select ename, sal, hiredate from emp order by ename; --asc 생략가능

select ename, sal, hiredate from emp order by ename desc;

--날짜로 오름차순 정렬
select hiredate from emp order by hiredate;

--급여로 내림차순 정렬
select sal from emp order by sal desc;

--1차 정렬과 2차 정렬, n차 정렬
select ename, deptno, sal, hiredate, job
from emp order by deptno asc, sal desc, job asc;


--student 테이블에서 아래 그림과 같이
-- 1전공이 102번인 학생들의 이름과 전화번호, 전화번호에서 국번 부분만 ***
--처리하여 출력하세요.
--결과
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

-- round(숫자, 출력할 자리수) 반올림함수
select round(987.12345, 4), round(987.12345, -1) from dual;

--update 데이터 수정
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

--update 데이터 수정
desc dept;
insert into dept values(40, 'OPERATIONS','SEOUL');

select*from dept;

--deptno가 10인 데이터의 'dname'을 'COUNTING'으로 수정
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

--40번 부서의 deptno를 50으로 변경
update dept
set deptno=50
where deptno=40;

select*from dept;

update dept
set deptno=40
where loc='BOSTON';


--job이 MANAGER인 사람의 COMM을 500로 수정
select*from emp;
update emp
set comm=500
where job='MANAGER';

--job이 salesman 이고 comm이 1000미만인 사람의 comm을 20% 인상
select*from emp;
update emp
set comm = comm*1.2
where job='SALESMAN' and COMM < 1000;
rollback;

--3월 16일
--trunc 절삭, 버림
select trunc(987.456,2),
trunc(987.456,0), trunc(987.456,-1) from dual;

--mod*** 나머지 함수
--ceil 주어진 숫자중 가장 가까운 정수(근)
--floor 주어진 숫자중 가장 가까운 정수(작은)
select mod(129,10)"mod",
ceil(123.45)"cell",
floor(123.45) "floor",
from dual;

--팀분류
select rownum"rownum", ceil

--power(숫자1, 숫자2) 숫자1의 승수
select power(2,4) from dual;

--날짜함수
--sysdate***

select sysdate from dual;
select to_char(sysdate,'yyyy-mm-dd')"오늘의 날짜" from dual;

--month_between(큰 수, 작은 수) 두 날짜 사이의 개월수 계산
select MONTHS_BETWEEN('14/09/30', '14/08/31') from dual;
select months_between(sysdate, hiredate) from emp;
select months_between(sysdate, hiredate)"months",
round(months_between(sysdate, hiredate),2)"months_round"from emp;

select months_between(sysdate, hiredate)"months",
to_char(round(months_between(sysdate,hiredate),2),'9999.99')"months_round" from emp;

--add_months
select sysdate, add_months(sysdate,1) from dual;

--next_day
select NEXT_DAY(sysdate, '월') from dual;

--last_day
select sysdate,last_day(sysdate) from dual;

--형변환(명시적 형변환(강제), 묵시적 형변환(자동))
--묵시적형변환
select 2+'3' from dual;
select 2+to_number('3')from dual;

select 2+'a' from dual;
select 2+ascii('a')from dual;
select ascii ('a')from dual;

select 2+'a' from dual;

--to_char
--yyyy-년도
--rrrr-년도 2000년 이후 년도
--year-연도의 영문이름

select sysdate,
to_char(sysdate,'yyyy')"yyyy",
to_char(sysdate, 'rrrr')"rrrr",
to_char(sysdate, 'yy')"yy",
to_char(sysdate, 'rr')"rr",
to_char(sysdate, 'year')"year"
from dual;

--mm 월자리로
--mon 유닉스용 오라클에서 사용, 윈도우에서는 month와 동일
--month 월을 뜻하는 전체 표기
select sysdate,
to_char(sysdate,'mm')"mm",
to_char(sysdate, 'mon')"mon",
to_char(sysdate, 'month')"month"
from dual;

--dd 일을 2자리로
--day 유닉스 오라클용, 윈도우에서 한글로
--ddth 몇 번째 날인지
select sysdate,
to_char(sysdate,'dd')"dd",
to_char(sysdate, 'day')"day",
to_char(sysdate, 'ddth')"ddth"
from dual;

--hh24 24시간제
--hh 12시간제
--mi 분
--ss 초
select sysdate, to_char(sysdate, 'rrrr-mm-dd:hh24:mi:ss')from dual;

select*from emp;

--to_char()
--9-9의 갯수만큼 자릿수 표시
--0- 빈자리를 0으로 채움
--$- $표시를 붙임
--.- 소수점 이하표시
--,- 천 단위 표시

select to_char (1234, '999999') from dual;
select to_char(1234, '099999') from dual;
select to_char(1234, '$9999') from dual;
select to_char(1234, '9999.99')from dual;
select to_char(12341234, '999,999,999')from dual;

--응용
select*from emp;
select ename,sal,comm, to_char((sal*12)+comm, '$999,999')"salary" from emp;

--응용문제
--professor 테이블을 조회하여 201번 학과에 근무하는 교수들의 이름과 
--급여,보너스,연봉을 아래와 같이 출력하세요. 단, 연봉은 (pay*12)+bonus
--로 계산합니다.

select*from professor;
select name, pay, bonus, to_char((pay*12)+bonus, '$999,999')"PAY"
from professor
where deptno=201;

--응용문제
--emp 테이블을 조회하여 comm 값을 가지고 있는 사람들의 empno, ename, hiredate,
--총연봉, 15%인상 후 연봉을 아래 화면 처럼 출력하세요. 단, 총연봉은 
--(sal*12)+comm으로 계산하고 아래 화면에서는 SAL로 출력되었으며
--15% 인상한 값은 총연봉의 15% 인상 값입니다.
--(HIREDATE 칼럼의 날짜 형식과 SAL 칼럼, 15% UP 칼럼의 $ 표시와, 기호 나오게 
--하세요

select*from emp;
select empno, ename, to_char(hiredate, 'yyyy-dd-mm')"HIREDATE"
, sal, to_char(((sal*12)+comm)*1.15, '$999,999')"15% 인상"
from emp
where comm is not null;

--to_number
select to_number('5')from dual;
select to_number('a')from dual; --오류남
select ascii('a') from dual; --문자 a를 아스키코드값으로 변형

--to_date
select to_date('2014/05/05') from dual;

--nvl(null value의 약자) *****
--nvl(col,0) col이 null이면 0으로 출력
select*from emp;

select ename, comm, nvl(comm,0) from emp;

--nvl 응용
-- professor 테이블에서 201번 학과 교수들의 이름과 급여, bonus,
-- 총 연봉을 아래와 같이 출력하세요. 단, 총 연봉은(pay*12+bonus)로
-- 계산하고 bonus가 없는 교수는 0으로 계산하세요.

select*from professor;
select name, bonus, pay, to_char(pay*12+nvl(bonus,0),'$999,999')"총 연봉"
from professor
where deptno = 201;

--nvl2(col1, col2, col3) col1이 null이 아니면 col2, col1이 null이면 col3
--nvl2 응용
--emp 테이블에서 deptno가 30번인 사람들의 empno, ename, sal, comm 값을
--출력하되 만약 comm 값이 null이 아니면 sal+comm 값을 출력하고 comm 값이
--null 이면 sal*0의 값을 출력하세요.
select empno, ename, sal, comm,
nvl2(comm,sal+comm,sal+0)
from emp;

--nvl2 응용
--아래 화면과 같이 emp 테이블에서 deptno가 30번인 
--사원들의 empno, ename, comm 조회하여
--comm 값이 있을 경우 'Exist'을 출력하고 comm 값이 null일 경우 'NULL'을 출력하세요.

select*from emp;
select empno, ename, comm, nvl2(comm, 'Exist', 'NULL')
from emp
where deptno=30;

--decode(a,b,'1',null) a가 b이면 1을 출력, null생략가능 --if함수********
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

--decode(a,b,decode(c,d,'1',null))a가 b일 경우에만 c가 d이면 1 출력 아닐 경우
--null.중첩decode
select deptno,name,decode(deptno,101,decode(name,'Audie Murphy','Best11'))"etc"
from professor;

select deptno,name,decode(deptno,101,decode(name,'Audie Murphy',
'Best11', 'N/A'))"etc"
from professor;

--case 활용
select*from student;
select name,tel
from student;

--tel에서 지역번호만 출력하는 칼럼을 tel2로 출력 ex) 051
select name, tel,
substr(tel,1,instr(tel,')')-1)tel2
from student;

--위 내용을 case에 적용하기
select name,tel,
case(substr(tel,1,instr(tel,')')-1)) when '02'then 'seoul' 
when '031' then 'gyeonggi'
when '051' then 'pusan'
when '031' then 'ulsan'
when '055' then 'gyeongnam'
else 'etc'
end tel2
from student;

--case 활용문제
student 테이블의 jumin 칼럼을 참조하여 학생들의 이름과 태어난 달, 
그리고 분기를 출력하세요. 태어난 달이 01-03월은 1/4, 04-06월은 2/4,
07-09월은 3/4, 10-12월은 4/4로 출력하세요.

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
--decode grade를 기준으로 name, grade, 학년 출력
--1-1학년
--2-2학년

select*from student;
select name, grade, decode(grade,1,'1학년',null)"1학년",
decode(grade,2,'2학년',null)"2학년",
decode(grade,3,'3학년',null)"3학년"
,decode(grade,'4학년',null)"4학년"
from student;

--관계설정에 관한 내용
create table one(
no number, name varchar2(10));

create table two(
num number, addr varchar2(30));

insert into two values(10,'SEOUL');
select*from two;
delete from two;

--제약조건 two 테이블에 추가 제약조건의 이름은 fk_num
alter table two add constraint fk_num foreign key(num) 
references one(no); --현재 불능

--one 테이블에 primary key 추가
alter table one add constraint pk_one_no primary key(no);

--one 테이블에 기본키를 추가 후 제약조건 two테이블에 추가 제약조건의 이름은
--fk_num
alter table two add constraint fk_num foreign key(num) references one(no);

--two테이블에 데이터 추가
insert into two values(10,'seoul'); --무결성 제약 조건위배,
-- Integrity constraint (HR.FK_NUM) violated

select*from one;
insert into one values(10,'blue');
insert into one values(20,'blue');

insert into two values(10,'seoul'); --one 테이블에 10번 레코드 추가후 실행 오케이
insert into two values(20,'pusan');

--제약조건 조회
select*from all_constraints
where table_name='TWO';

--제약조건 삭제
alter table two drop constraint fk_num;

insert into two values(30,'seoul');

insert into two values(10,'seoul'); -- one 테이블에 10번 레코드 추가후 실행 오케이
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

--emp 테이블에 primary 키 추가하기
alter table emp add constraint pk_emp_empno primary key(empno);

--emp 테이블에 제약조건 추가하기
alter table emp add constraint fk_empno foreign key(empno) 
references emp(empno);


--2021/03/17 실습
create table rental(HAKBUN number(20),NAME varchar2(20),CITY varchar(20)
,LOCKER number);

select*from rental;
select*from lockerroom;

select*from all_constraints where table_name='rental';
select*from all_constraints where table_name='lockerroom';

--lockerroom 테이블에 기본키를 추가후 제약조건 rental테이블에 추가 
--제약조건의 이름은 fk_locker
alter table rental add constraint fk_locker foreign key(locker) 
references lockerroom(no);

insert into rental values(1111,'전지현','서울',1);
insert into rental values(2222,'구혜선','대구',2);
insert into rental values(3333,'송혜교','부산',3);

create table lockerroom(no number primary key, name varchar2(50));
insert into lockerroom values(1,'1번 사물함');
insert into lockerroom values(2, '2번 사물함');
insert into lockerroom values(3, '3번 사물함');

--그룹함수
--count() 갯수
select*from emp;
select count(comm), count(mgr), count(*) from emp;

--sum()
select count(comm), sum(comm)"sum" from emp;

--avg() 평균 -- 주의 사항 null을 제외한 평균, 전체평균
select count(comm), sum(comm), avg(comm) from emp;
select count(comm)"comm 인원", count(*)"총인원",sum(comm),avg(comm),
sum(comm)/count(*)"전체평균" from emp;
--avg가 총인원에 대한 평균이 아니라 comm의 인원에 대한 총편균을 의미한다.
--전체 평균을 구하고 싶으면 sum/count(*)를 사용하면 된다.
-- 혹은 avg(nvl(comm,0))을 이용할 수도 있다.


--일반함수들 중복해서 사용하기 전체평균
select count(comm)"comm인원",count(*)"총인원", sum(comm), avg(comm), 
round(avg(nvl(comm,0)),2)"전체평균" from emp;

--max min
select max(sal), min(sal)
from emp;

select max(hiredate)"max",
min(hiredate)"min"
from emp;

--stddev 표준편차, variance 분산
select round(stddev(sal),2),
round(variance(sal),2)
from emp;

--그룹함수 같은 경우는 group by가 먼저 이루어져야 출력가능
-- 아래와 같은 출력 순서 때문이다.
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

--중첩그룹
select deptno, job, avg(sal)"avgsal"
from emp
group by deptno, job
order by deptno, job asc;

--주의 select 절에서 사용된 칼럼은 group화 시켜야 한다. 
select job,deptno,avg(sal)"avgsal"
from emp
group by deptno, job
order by 1,2;

--having 그룹의 조건
select*from emp;
select deptno, avg(nvl(sal,0))
from emp
where avg(nvl(sal,0))>2000; --오류

select deptno,avg(nvl(sal,0))
from emp
group by deptno
having avg(nvl(sal,0))>=2000;

select*from professor;

--deptno를 별로 count 와 pay 합을 출력
select*from professor;
select deptno,count(*),sum(pay)
from professor
group by deptno;

--emp테이블에서 2005년 부서별 사원수 출력
select '2005년',deptno"부서번호", count(*)"사원수"
from emp
group by deptno;

--부서별로 그룹하여 부서번호, 인원수, 급여의 평균, 급여의 합을 조회
select*from emp;
select deptno, count(*), avg(sal), sum(sal)
from emp
group by deptno;

--업무별로 그룹하여 업무, 인원수,평균 급여액, 최고 급여액, 최저 급여액 및 합계 조회
select*from emp;
select job, count(*)"인원수", round(avg(sal),2)"평균",
max(sal)"최고임금액", min(sal)"최저임금액", sum(sal)"급여합계"
from emp
group by job;

--having 문제
--사원 수가 다섯 명이 넘는 부서와 사원 수를 조회
select*from emp;
select deptno, count(*)"사원수"
from emp
group by deptno
having count(*)>5;

--전체 월급이 5000을 초과하는 JOB에 대해서 JOB과 월급이 합계를 조회하는 예이다.
--단 판매원(SALES)은 제외하고 월 급여 합계로 내림차순 정렬

select*from emp;
select job,sum(sal)"급여합계"             
from emp
where job !='SALESMAN'
group by job
having sum(sal)>5000
order by sum(sal) desc;

create table 국가(국가 varchar2(20));
insert into 국가 values('국가코드');
insert into 국가 values('국가명');
insert into 국가 values('화폐명');
select*from 국가;

create table 사용자(사용자 varchar2(20));
insert into 사용자 values('ID');
insert into 사용자 values('이름');
insert into 사용자 values('국가코드');
insert into 사용자 values('나이');
insert into 사용자 values('성별');
select*from 사용자;


alter table 국가 add constraint fk_country primary key(국가);
alter table 사용자 add constraint fk_user primary key(사용자);


select*from all_constraints
where table_name='국가';

select*from all_constraints
where table_name='사용자';

---오라클 문제 연습
--1번 문제
select*from emp;
select empno, ename, sal
from emp
where deptno = 10;

--2번 문제
select*from emp;
select ename, hiredate, deptno
from emp
where deptno = 7369;

--3번 문제
select*from emp where ename='ALLEN';

--4번 문제
select*from emp;
select ename, deptno, sal 
from emp
where hiredate='83/01/12';

--5번 문제
select*from emp;
select*from emp where job != 'MANAGER';

--6번 문제
select*from emp where hiredate > '81/04/02';

--7번 문제
select ename,sal,deptno
from emp
where sal > 800;

--8번문제
select*from emp where deptno >= 20;

--9번문제
select*from emp where ename > 'K';

--10번문제
select*from emp where hiredate < '81/12/09';

--11번 문제
select empno,ename
from emp
where empno <= 7698;

--12번 문제
select ename, sal, deptno
from emp
where hiredate > '81/04/02' and hiredate < '82/12/09';

--13번 문제
select ename, job, sal
from emp
where sal > 1600 and sal < 8000;

--14번 문제
select*from emp where empno >= 7654 and empno <= 7782;

--15번 문제
select*from emp where ename >= 'B' and ename <='J';

--16번 문제
select*from emp where hiredate > '82/01/01' or hiredate < '80/12/31';

--17번 문제
select*from emp where job='MANAGER' or job='SALESMAN';

--18번 문제
select ename, empno, deptno
from emp
where deptno != 20 and deptno != 30;

--19번 문제
select*from emp where ename like 'S_%';

--20번 문제
select*from emp where hiredate >='81/01/01' and hiredate <= '81/12/31';

--21번 문제
select*from emp where ename like('%S%');

--22번 문제
select*from emp where ename like 'S___T';

--23번 문제
select*from emp where ename like '_A%';

--24번 문제
select*from emp where COMM is null;

--25번 문제
select*from emp where COMM is NOT null;

--26번 문제
select ename,deptno, sal
from emp
where deptno=30 and sal>1500;

--27번 문제
select ename, empno, deptno
from emp
where ename like 'K_%' or deptno=30;

--28번 문제
select*from emp where sal > 1500 and deptno=30 and job='MANAGER';

commit;

--3/17 복습
--sum() 합계
select count(comm), sum(comm)"sum" from emp;

--avg()평균
select count(comm),sum(comm),avg(comm) from emp;
select count(comm)"comm인원",count(*)"총인원", sum(comm),avg(comm) from emp;




