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

--decode(a,b,'1',null) a가 b이면 1을 출력, b가 아니면 null을 출력
--null생략가능 --if함수********
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

--20210318 수업
-- join
-- 테이블 만들기
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

--inner join 교집합, 왼쪽 테이블값을 기준으로 오른쪽 테이블에 일치하는 데이터 조회
select a.id, name, car
from tbl1 a inner join tbl2 b 
on a.id=b.id;  --ansl style

select a.id,a.name,b.car
from tbl1 a, tbl2 b
where a.id=b.id; --oracle style

--left outer join
--왼쪽 요소는 다 보여주고 오른쪽꺼는 왼쪽과 일치하는 내용 출력
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
--오른쪽 요소는 다 보여주고 왼쪽꺼는 오른쪽과 일치하는 내용 출력
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


--left join을 이용해서 inner join을 표현
select  a.id,a.name,b.car
from tbl1 a left outer join tbl2 b
on a.id=b.id
where b.id is not null;


--right join을 이용해서 ineer join을 표현
select  a.id,a.name,b.car
from tbl1 a right outer join tbl2 b
on a.id=b.id
where a.id is not null; 

--미션완료

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

--카티션곱 -경우의 수 모두 출력 조건을 생략한 상태
select a.name, b.name, c.name
from cat_a a, cat_b b,  car_c c;

--a b c의 no를 등가 조인
select a.name, b.name, c.name
from cat_a a, cat_b b, cat_c c
where a.no=b.no
and b.no=c.no;

--3개의 테이블 조회에서 조인조건절에 2개 테이블만 조건을 넣은 카티션곱
select a.name, b.name, c.name
from cat_a a, cat_b b, cat_c c
where a.no=b.no;

--등가조인(equl join)
--조건절에서 연산자 =을 사용한 조인
--emp 테이블 활용
select empno, ename, dname
from emp e, dept d
where e.deptno  = d.deptno;

select empno, ename, dname
from emp e join dept d
on e.deptno=d.deptno;

--조인 사용에
--학생 테이블(student) 과 교수 테이블(professor)을
--join 하여 학생의 이름과 지도교수번호, 지도교수 이름을 출력하세요.
select*from student;
select*from professor;

select  s.name, p.profno, p.name
from student s, professor p
where  s.profno= p.profno;

--student  테이블에서 1전공이 101번 학생의 이름과 각 학생들의 지도교수를 출력
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

--비등가 조인(Non-Equl join)
--Customer 테이블과 gift 테이블을 Join하여 고객별로 마일리지 포인트를 조회한 후
--해당 마일리지 점수로 받을 수 있는 상품을 조회하여 고객의 이름과 받을 수 있는 상품 명을 
--아래와 같이 출력하세요.

--ORACLE JOINT
select*from customer;
select*from gift;
select c.gname "CUST_NAME",to_char(c.point,'999,999')"POINT", g.gname"GIFT_SET"
from customer c, gift g
where c.point between g.g_start  and g.g_end;

--JOIN ON 사용해서 ANSI JOIN
select c.gname"CUST_NAME",to_char(c.point,'999,999')"POINT",g.gname"GIFT_SET"
from customer c join gift g
on c.point between g.g_start and g.g_end;

--위 내용을 비교 연산자로 변형
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

--student 테이블에서 score, hakjum 테이블을 조회
--학생이름 점수 학점을 조회
select*from student;
select*from hakjum;
select*from score;

--student 테이블과 score, hakjum 테이블을 조회
--학생이름 점수 학점을 조회
--결과화면
--name  점수  학점
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
--CLARK의 상사 KING 입니다.
select e1.ename, e2.ename
from emp e1, emp e2
where e1.mgr=e2.empno;

select  e1.ename"사원이름", e2.ename"매니저이름"
from emp e1, emp e2
where e1.mgr=e2.empno;-- ORACLE JOIN

select e1.ename, e2.ename
from emp e1 join emp e2
on e1.mgr=e2.empno;  --ANSI JOIN

--SUBQUERY 쿼리 속에 쿼리
--사용하는 곳
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

--emp WARD 의 COMM보다 작은 사람의 이름과 커미션 출력
select ename, comm
from emp
where comm < (select comm from emp where ename='WARD');

--서브쿼리의 주의점
--1. SUBQUERY 부분은 WHERE 절에 연산자 오른쪽에 위치하면 반드시 (  )로 묶어야 한다.
--2. 특별한 경우를 제외하고 SUBQUERY 절에는 ORDER BY를 사용할 수 없다.
--3. 단일행 SUBQUERY, 다중행 SUBQUERY의 연산자를 잘 구분하여야 한다.

--단일행 서브쿼리
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

--student 테이블과 department 테이블을 사용하여 'Anthony Hopkins' 학생과
-- 1전공(deptno1)이 동일한 학생들의 이름과 1전공 이름을 출력

select*from student;
select*from department;

select s.name,d.dname
from student s, department d
where s.deptno1 =  d.deptno
and s. deptno1= 
(select deptno1 from student where name='Anthony Hopkins'); 

--student 테이블에서 1전공(deptno1)이 201번인 학과 평균 몸무게보다 몸무게가 많은
--학생의 이름과 몸무게 출력
--1. student 테이블에서 1전공(deptno1)이 201번 먼저 확인

select name, weight
from student
where weight > (select avg(weight) from student where deptno1 = 201);

-------------------------------
--파일이름: sql318_1_40  윤요섭.txt
--카톡전송필
--------------------------------
select*from emp;
--29번 문제
select  empno from emp where deptno = 30;

--30번 문제
select *from  emp  order by  sal desc;

--31번 문제
select*from emp order by deptno asc, sal desc;

--32번 문제
select*from emp order by deptno desc, ename asc, sal desc;

--33번 문제
select ename, sal, comm, sal+comm"총액" from emp order by sal desc;

--34번 문제
select ename, sal, sal*1.13"bonus", deptno
from emp;

--35번 문제
select ename, deptno, sal, sal*11+sal*1.5"연봉"
from emp where deptno=30;

--36번 문제
select ename, sal, round(sal/12/5,1)" 시간당 임금" from emp where deptno=20; 

--37번 문제
select ename, sal, round(sal*0.15,2)" 회비" from emp where sal >= 1500 and sal <= 3000;

--38번 문제
select ename, sal, sal*0.15"경조비" from emp where sal >= 2000;

--39번 문제 --도저히 모르겠음
select*from emp;
select  deptno, ename, hiredate, sysdate, date(between  hiredate and sysdate)"근무일수",
date((between  hiredate and sysdate)/365)"근무년수", date((between  hiredate and sysdate)/365*12)"근무월수",
date((between  hiredate and sysdate)/7)"근무주수"
from emp;

--40번 문제
select ename, sal, sal-sal*0.1"실수령액"   from emp order by sal desc;

------------------------------

commit;

--2021/03/19 -- hruser 접속 안 됨 향후 ;뒤에 전부 컨트롤 엔터 누를 것
-- 단중행 SUBQUERY
--다중행 쿼리 연산자
--/*****
-- exists 값이 있으면 메인 수행
-- > any 최소값
-- < any 최대값
-- < all 최소값
-- > all 최대값
-- *****/


CREATE TABLE DEPT2 (
 DCODE   VARCHAR2(06)  PRIMARY KEY,
 DNAME   VARCHAR2(30) NOT NULL,
 PDEPT VARCHAR2(06) ,
 AREA        VARCHAR2(30)
);

INSERT INTO DEPT2 VALUES ('0001','President','','Pohang Main Office');
INSERT INTO DEPT2 VALUES ('1000','Management Support Team','0001','Seoul Branch Office');
INSERT INTO DEPT2 VALUES ('1001','Financial Management Team','1000','Seoul Branch Office');
INSERT INTO DEPT2 VALUES ('1002','General affairs','1000','Seoul Branch Office');
INSERT INTO DEPT2 VALUES ('1003','Engineering division','0001','Pohang Main Office');
INSERT INTO DEPT2 VALUES ('1004','H/W Support Team','1003','Daejeon Branch Office');
INSERT INTO DEPT2 VALUES ('1005','S/W Support Team','1003','Kyunggi Branch Office');
INSERT INTO DEPT2 VALUES ('1006','Business Department','0001','Pohang Main Office');
INSERT INTO DEPT2 VALUES ('1007','Business Planning Team','1006','Pohang Main Office');
INSERT INTO DEPT2 VALUES ('1008','Sales1 Team','1007','Busan Branch Office');
INSERT INTO DEPT2 VALUES ('1009','Sales2 Team','1007','Kyunggi Branch Office');
INSERT INTO DEPT2 VALUES ('1010','Sales3 Team','1007','Seoul Branch Office');
INSERT INTO DEPT2 VALUES ('1011','Sales4 Team','1007','Ulsan Branch Office');

commit;

CREATE TABLE EMP2 (
 EMPNO       NUMBER  PRIMARY KEY,
 NAME        VARCHAR2(30) NOT NULL,
 BIRTHDAY    DATE,
 DEPTNO      VARCHAR2(06) NOT NULL,
 EMP_TYPE    VARCHAR2(30),
 TEL         VARCHAR2(15),
 HOBBY       VARCHAR2(30),
 PAY         NUMBER,
 POSITION    VARCHAR2(30),
 PEMPNO      NUMBER
);

INSERT INTO EMP2 VALUES (19900101,'Kurt Russell',TO_DATE('19640125','YYYYMMDD'),'0001','Permanent employee','054)223-0001','music',100000000,'Boss',null);
INSERT INTO EMP2 VALUES (19960101,'AL Pacino',TO_DATE('19730322','YYYYMMDD'),'1000','Permanent employee','02)6255-8000','reading',72000000,'Department head',19900101);
INSERT INTO EMP2 VALUES (19970201,'Woody Harrelson',TO_DATE('19750415','YYYYMMDD'),'1000','Permanent employee','02)6255-8005','Fitness',50000000,'Section head',19960101);
INSERT INTO EMP2 VALUES (19930331,'Tommy Lee Jones',TO_DATE('19760525','YYYYMMDD'),'1001','Perment employee','02)6255-8010','bike',60000000,'Deputy department head',19960101);
INSERT INTO EMP2 VALUES (19950303,'Gene Hackman',TO_DATE('19730615','YYYYMMDD'),'1002','Perment employee','02)6255-8020','Marathon',56000000,'Section head',19960101);
INSERT INTO EMP2 VALUES (19966102,'Kevin Bacon',TO_DATE('19720705','YYYYMMDD'),'1003','Perment employee','052)223-4000','Music',75000000,'Department head',19900101);
INSERT INTO EMP2 VALUES (19930402,'Hugh Grant',TO_DATE('19720815','YYYYMMDD'),'1004','Perment employee','042)998-7005','Climb',51000000,'Section head',19966102);
INSERT INTO EMP2 VALUES (19960303,'Keanu Reeves',TO_DATE('19710925','YYYYMMDD'),'1005','Perment employee','031)564-3340','Climb',35000000,'Deputy Section chief',19966102);
INSERT INTO EMP2 VALUES (19970112,'Val Kilmer',TO_DATE('19761105','YYYYMMDD'),'1006','Perment employee','054)223-4500','Swim',68000000,'Department head',19900101);
INSERT INTO EMP2 VALUES (19960212,'Chris O''Donnell',TO_DATE('19721215','YYYYMMDD'),'1007','Perment employee','054)223-4600',null,49000000,'Section head',19970112);
INSERT INTO EMP2 VALUES (20000101,'Jack Nicholson',TO_DATE('19850125','YYYYMMDD'),'1008','Contracted Worker','051)123-4567','Climb', 30000000,'',19960212);
INSERT INTO EMP2 VALUES (20000102,'Denzel Washington',TO_DATE('19830322','YYYYMMDD'),'1009','Contracted Worker','031)234-5678','Fishing', 30000000,'',19960212);
INSERT INTO EMP2 VALUES (20000203,'Richard Gere',TO_DATE('19820415','YYYYMMDD'),'1010','Contracted Worker','02)2345-6789','Baduk', 30000000,'',19960212);
INSERT INTO EMP2 VALUES (20000334,'Kevin Costner',TO_DATE('19810525','YYYYMMDD'),'1011','Contracted Worker','053)456-7890','Singing', 30000000,'',19960212);
INSERT INTO EMP2 VALUES (20000305,'JohnTravolta',TO_DATE('19800615','YYYYMMDD'),'1008','Probation','051)567-8901','Reading book', 22000000,'',19960212);
INSERT INTO EMP2 VALUES (20006106,'Robert De Niro',TO_DATE('19800705','YYYYMMDD'),'1009','Probation','031)678-9012','Driking',   22000000,'',19960212);
INSERT INTO EMP2 VALUES (20000407,'Sly Stallone',TO_DATE('19800815','YYYYMMDD'),'1010','Probation','02)2789-0123','Computer game', 22000000,'',19960212);
INSERT INTO EMP2 VALUES (20000308,'Tom Cruise',TO_DATE('19800925','YYYYMMDD'),'1011','Intern','053)890-1234','Golf', 20000000,'',19960212);
INSERT INTO EMP2 VALUES (20000119,'Harrison Ford',TO_DATE('19801105','YYYYMMDD'),'1004','Intern','042)901-2345','Drinking',   20000000,'',19930402);
INSERT INTO EMP2 VALUES (20000210,'Clint Eastwood',TO_DATE('19801215','YYYYMMDD'),'1005','Intern','031)345-3456','Reading book', 20000000,'',19960303);
COMMIT;


--EMP2 테이블과 DEPT2 테이블을 참조하여
--근무지역(dept2 테이블의 area 칼럼)이 'Pohang Main Office'인 
--모든 사원들의 사번과 이름, 부서번호를 출력하세요.

--1'Pohang Main Office'인 decode를 출력

select dcode
from dept2
where area='Pohang Main Office';

select empno, name, deptno
from emp2
where deptno in(select dcode
from dept2
where area='Pohang Main Office');

--exists
select deptno
from dept2
where deptno=20;

select*from dept where exists(select deptno
from dept
where deptno=&no);

--in 연산자와 비교
select*from dept
where deptno in (select deptno
from dept
where deptno=&no);

--문제해결
--emp2테이블에서 전체 직원중 "Section head"직급의 최소 연봉자보다 연봉이
--높은 사람의 이름과 직급, 연봉을 출력
--단 연봉은 $100,000,00 형식으로 출력하시오
--1. emp2 테이블에서 전체 직원중 "Section head"의 연봉

select pay
from emp2
where position='Section head';

select name, position, pay
from emp2
where pay > any(select pay
from emp2
where position = 'Section head');

select name,position, pay
from emp2
where pay > any(50000000,
56000000,
51000000,
49000000);

--문제2
--student에서 전체 학생 중에서 체중이
--2학년 학생들의 체중에서 가장 적게 나가는 학생보다 몸무게가
--적은 학생의 이름과 학년과 몸무게 출력
-- 1.        2학년 학생들의 체중에서

select weight
from student
where grade=2;

select name, grade, weight
from student
where weight < all(select weight
from student
where grade=2);

--문제3
--emp2 테이블과 dept2테이블에서 각 부서별 평균 연봉을 구하고
--그 중에서 평균 연봉이 가장 적은 부서의 평균 연봉보다 적게 받는
--직원들의 부서명, 직원명 연봉 출력
-- 1.   평균 연봉이 가장 적은 부서의 평균 연봉
select*from emp2;
select*from dept2;

select avg(pay)
from emp2
group by deptno
order by 3;
--order by avg(pay) desc; --참고자료

select d.dname,e.name.to_char(e.pay,'$999,999,999')"salary"
from emp2 e, dept d
where e.deptno = d.code
and e.pay<all(select avg(pay)
from emp2
group by deptno)
order by 3;

--단중 칼럼 SUBQUERY
--서브쿼리의 결과값이 여러 칼럼인 경우에 적용
select*from student;
--학년별로 최대몸무게를 가진 학년과 최대 몸무게 출력
select grade, max(weight)
from student
group by grade;
--위서브쿼리의 결과값에 해당(매칭되는)하는 학년,이름,몸무게 출력
--in연산자를 사용해서 한줄 한줄 메인 쿼리로 적용
--학년별로 몸무게가 제일 많은 학생의 이름과 몸무게를 표현
select grade, name, weight
from student
where (grade, weight, aaa) in (select grade, max(weight)
from student
group by grade);

--다중칼럼 문제
--professor, department테이블에서 각 학과별로
--입사일이 가장 오래된 교수의
--교수 번호와 이름, 학과명 출력 (입사일 순으로 오름차순 정렬)

select to_char(min(hiredate),'yyyy-mm-dd') from professor;
--1.  입사일이 학과별로 가장 적은 교수번호와 입사일을 출력
select deptno, min(hiredate)
from professor
group by deptno;

--select p.profno 
--from professor p, department d
--where p.deptno=d.deptno
--and (p.deptno,p.hiredate) in (서브쿼리);

select p.profno, p.name, d.dname, p.hiredate
from professor p, department d
where p.deptno=d.deptno
and (p.deptno,p.hiredate) in (select deptno, min(hiredate)
from professor group by deptno)
order by p.hiredate;

--다중칼럼 문제2
--emp2 테이블에서 직급별로 해당 직급에서 최대 연봉을 받는 지원의 이름과
--직급, 연봉을 출력, 연봉기준 오름차순
--order by 3은 3번째 칼럼은 오름차순대로 정렬한다는 의미


--1. 직급별로 최대 연봉구하기
select position,max(pay) from emp2 group by position;


--2. subquery 적용하기
select name, position, pay
from emp2
where (position, pay)  in(select position,max(pay) from emp2 group by position)
order by 3;

--정규화(Normalization)
select*from emp;
select*from dept;

select e.ename, e.job, e.deptno, d.dname "부서이름"
from emp e, dept d
where e.deptno=d.deptno;

--view
--데이터 관리용으로 view
--일반적으로 주로 데이터를 조회하는 목적으로 사용됩니다.

--비전에 따라서 시스템에서 권한 주기 필요할 수도 있음
--system > grant create view to scott;

show user;

create or replace view v_emp1
as
select empno,ename, hiredate
from emp;

select*from v_emp1;

create index idx_v_emp_ename
on v_emp1(ename);

--view 삭제
drop view v_emp1;

--view는 가상 테이블이고, view에는 실제 데이터가 없으며,
--데이터를 불러오는 query 만 존재

--단순 view
--서브쿼리 조건이 들어가지 않고 한 개의 테이블로 구성된 뷰

create table o_table(
a number, b number);

create view view1
as
select a,b
from o_table;

select*from view1;
insert into view1 values(1,2);

select*from view1;
select*from o_table;
rollback;

create view view2
as
select a,b
from o_table
with read only;
select*from view2;


insert into view1 values(3,4);

select*from view2;
select*from o_table;

create view view3
as
select a,b
from o_table
where a=3
with check option;

--b가 4인 값을 a를 5로 변경
update view3
set a=5
where b=4; --오류 (해당 칼럼만 변경이 가능)

--a가 3일때 b를 6으로 변경
update view3
set b=6
where a=3;

select*from view3;

--view 활용하기

create view v_empdept
as
select empno, ename, dname
from emp e, dept d
where e.deptno=d.deptno;

select*from v_empdept;

--미션
--nice 사용자 만들어 권한주기
--system에서 view 생성권한 주기
--제공테이블 만들어 관계설정하고
--join을 하고
--join 한 결과를 v_test라는 이름으로 만들기

--CMD로 하는 법
--SQL > conn/ as sysdba
--SQL > show user;
-- USER is "SYS"
--SQL > grant create view to red;
--Grant succeeded

--view_윤요섭.txt 로 월요일까지 보낼것

create table 구매자정보(구매자번호 number, 상품번호 number,
 구매자아이디 varchar2(50), 구매자한글이름 varchar2(50), 구매자영문이름 varchar2(50),
구매자성별 varchar2(50), 구매자연락처 number, 구매자이메일 varchar2(50), 등록일 date);

insert into 구매자정보 values(1, 111000, 'gdragon', '권지용',' kwonjiyong', '남', 01045095102, 'gd11', '15/12/07');
insert into 구매자정보 values(2, 111001, 'jenni', '김제니', 'kimjenni', '여', 01055862296, 'jenni22', '16/01/29');
insert into 구매자정보 values(3, 121000, 'IU', '이지은', 'leejieun','여',01041825437, 'iu33', '16/05/17');
insert into 구매자정보 values(4, 121001, 'taenggu', '김태연','kimtaeyeon','여',01054682216,'taeyeon44','17/05/16');
insert into 구매자정보 values(5, 131000, 'jaedragon', '이재용', 'leejaeyong','남',01091045526, 'jd55', '17/08/09');
insert into 구매자정보 values(6, 131001, 'psy', '박재상','parkjaesang','남',01071076106, 'psy66', '17/08/17');

select*from 구매자정보;

create table 상품정보(상품번호 number, 상품명 varchar2(50),
상품간략소개 varchar2(50), 상품가격 number, 등록일 date); 
insert into 상품정보 values(111000, '디올','너무비쌈', 95021980, '15/12/17' );
insert into 상품정보 values(111001, '샤넬', '완전비쌈', 100000000, '16/01/29');
insert into 상품정보 values(121000, '롤스로이스','절대못삼',796000000,'16/05/17');
insert into 상품정보 values(121001, '람보르기니','후덜덜',699000000, '17/05/16');
insert into 상품정보 values(131000,  '삼성','끝판왕',123456789123456789, '17/08/09');  
insert into 상품정보 values(131001, '프라다','백',1234565798, '17/08/17');
select*from 상품정보;

alter table 구매자정보 add constraint fk_상품번호_num foreign key(상품번호) references 상품정보(상품번호);
alter table 구매자정보 add constraint pk_구매자번호_no primary key(구매자번호);
alter table 상품정보 add constraint  pk_상품번호_no primary key(상품번호);

select a.구매자번호, a.상품번호, a.구매자한글이름, a.구매자영문이름,
a.구매자성별, a.구매자연락처, a.구매자이메일, a.등록일,  b.상품명, b.상품간략소개,
b.상품가격
from 구매자정보 a, 상품정보 b
where a.상품번호=b.상품번호;

create view v_test
as
select a.구매자번호, a.상품번호, a.구매자한글이름, a.구매자영문이름,
a.구매자성별, a.구매자연락처, a.구매자이메일, a.등록일,  b.상품명, b.상품간략소개,
b.상품가격
from 구매자정보 a, 상품정보 b
where  a.상품번호=b.상품번호;

select*from v_test;

commit;

--view 문제
--v_stu studno, name, jumin, tel, profno의 내용을
-- 뷰이름: v_stu 칼럼: studn, name, jumin, tel, profno의 내용
select*from student;
create or replace view v_stu 
as
select studno, name, jumin, tel, profno
from student;

select*from v_stu;

--subquery 문제
--emp2 테이블에서 hobby별로 최대 연봉을 받은 직원의 이름과 직급, 연봉을
--출력, 연봉을 기준으로 오름차순
select*from emp2;
--1. hobby 별로 최대연봉
select hobby, max(pay)
from emp2
group by hobby;

select name, hobby, pay
from emp2
where (hobby, pay) in (select hobby, max(pay)
from emp2
group by hobby)
order by 3;

--3/22 수업
-- view 이어서
-- 복합뷰
-- subquery 부분에 여러 개의 테이블을 조인한 결과 값
--주의점 여러테이블을 조인해서 사용하면 성능 저하의 원인이 될 수 있습니다.

create or replace view v_emp
as
select e.ename, d.dname
from emp e, dept d
where e.deptno=d.deptno;

select*from v_emp;

--인라인 뷰(inline view)
-- 뷰의 의미는 여러번 반복해서 사용인데, 인라인뷰는 일회성입니다.
-- 사용 예) emp, dept테이블을 이용 부서번호 
-- 부서별 최대 급여 및 부서명을 출력 (inline view로 처리)

-- 1. 부서별 최대급여 출력
select deptno, max(sal), sal
from emp
group by deptno, sal;

--2.lnline view로 처리
select e.deptno, d.dname, e.sal
from (select deptno, max(sal)sal from emp group by deptno) e, dept d
where e.deptno=d.deptno;  --sal가 대문자일때는 가능 소문자는 에러

-- 결과값은 같은나 서브쿼리시 select안에서는 sal을 붙여줘야함
select deptno, sal from emp group by deptno, sal;
select deptno, max(sal)sal from emp group by deptno, sal;
select deptno, max(sal)"SAL" from emp group by deptno, sal;


select*from emp;

--lag와 inline을 결합


--lag(출력할 칼럼, offset, 기본출력)
-- over: order by, group by 편리하게 개선
-- lag over를 통해 한 칸씩 칼럼을 뒤로 미룰 수 있음
select*from professor;
select name, deptno, lag(deptno) over(order by deptno) from professor;

--앞 튜플과의 차를 구할 때 사용할 수 있음
select name, deptno, lag(deptno,1,0) over(order by deptno),
deptno-(lag(deptno,1,0) over(order by deptno)) diff from professor;

-- 1. inline view로 사용되는 내용 출력
select*from professor;
select lag(deptno) over(order by deptno)ndeptno,deptno,profno,name
from professor;

-- 2. deptno가 같으면 출력이 되지 않도록, 다르면 출력되도록
select decode(deptno, ndeptno, '', deptno)deptno, profno, name
from (select lag(deptno) over(order by deptno)ndeptno,deptno,profno,name
from professor);

-- view 조회, 삭제
select*from user_views;

select view_name,text,read_only
from user_views;

drop view v_emp1;

--end view


--기존 테이블 구조 변형
create table aaa(id number, name varchar2(10));
--테이블 이름변경
alter table aaa rename to bbb;
desc aaa; -- 변경후 존재하지 않음
desc bbb;

alter table bbb rename to aaa;
desc bbb;
desc aaa;

-- 칼럼 추가 
alter table aaa add(addr varchar2(50));
desc aaa;

--칼럼 삭제 
alter table aaa drop column addr;
desc aaa;

--데이터 추가
insert into aaa values(1, 'hong', 'seoul');
select*from aaa;

--칼럼 이름변경
alter table aaa rename column addr to memo;
desc aaa;

--칼럼의 타입변경
alter table aaa modify(memo char(5));
desc aaa;

--데이터가 있는 타입변경2
alter table aaa modify(memo number);
select*from aaa;
desc aaa;
delete from aaa;
insert into aaa values(1,'hong',111);
select*from aaa;

--sequence
--연속적인 일련번호를 자동으로 부여하는 기능

create SEQUENCE jno_seq
increment by 1
start with 100
maxvalue 110
minvalue 90
cycle
cache 2;

create table s_order(
ord_no number(4),
ord_name varchar2(10),
p_name varchar2(20),
p_qty number);

--nextval은 자동으로 1씩 증가하게끔 함
--currval은 자동으로 현재 숫자를 복사 및 붙여넣기 함
--maxvalue을 초과하면 minvalue부터 다시 시작
select jno_seq.nextval from dual;
select jno_seq.nextval from dual;


insert into s_order values(jno_seq.nextval,'james','apple',5);
select*from s_order;
insert into s_order values(jno_seq.currval,'james','apple',5);

--procedure를 활용한 입력
begin
for i in 1..20loop
 insert into s_order values(jno_seq.nextval,'james','apple',5);
end loop;
commit;
end;
/
select*from s_order;

--감소하는 sequence 만들기
create sequence jno_seq_rev
increment by -2
minvalue 0
maxvalue 20
start with 10;

--테이블 만들기
create table s_rev1(no number);

insert into s_rev1 values(jno_seq.nextval);
select*from s_rev1;
delete from s_rev1;

--시퀀스 수정 cycle, nocycle
alter sequence jno_seq_rev
increment by 5
cache 2 --저장공간부여
nocycle;

-- 시퀀스의 초기화
create table board (no number primary key,
title varchar2(100),
content varchar2(100),
writter varchar2(30),
wday date);
create sequence board_seq;

insert into board values(board_seq.nextval, 'tit1','conts1','h1',sysdate);
select*from board;


CREATE OR REPLACE PROCEDURE re_seq
(
   SNAME IN VARCHAR2
)
IS
   VAL NUMBER;
BEGIN
   EXECUTE IMMEDIATE 'SELECT ' || SNAME || '.NEXTVAL FROM DUAL ' INTO  VAL;
   EXECUTE IMMEDIATE 'ALTER SEQUENCE ' || SNAME || ' INCREMENT BY -' || VAL ||
   ' MINVALUE 0';
   EXECUTE IMMEDIATE 'SELECT ' || SNAME || '.NEXTVAL FROM DUAL ' INTO VAL;
   EXECUTE IMMEDIATE 'ALTER SEQUENCE ' || SNAME || ' INCREMENT BY 1 MINVALUE 0';
END;
/

--프로시저 실행
exec re_seq('BOARD_SEQ');

select*from board;
select board_seq.currval from dual;

--t_seq 만들기
create sequence t_seq
minvalue 0
start with 1;


--시퀀스 조회
select*from user_sequences
where sequence_name='BOARD_SEQ';

--시퀀스삭제
drop sequence board_seq;

--procedure
-- pl/sql: procedual language
-- pl/sql을 이용해서 데이터베이스에 실행절차를 저장해서 반복사용
-- 구조:DECLARE
--선언: BEGIN
--실행: EXCEPTION
--예외: END
*/

SET SERVEROUTPUT ON;
/
BEGIN
 DBMS_OUTPUT.PUT_LINE('HELLO BIG DATA');
 END;
/
--변수에 할당 후에 내용 출력
CREATE OR REPLACE PROCEDURE HELLO_BIG
IS
  I_MESSAGE VARCHAR2(100):='HELLO_BIG DATA'; --할당연산(대입연산)
BEGIN
 DBMS_OUTPUT.PUT_LINE(I_MESSAGE);
END; 
/
EXEC HELLO_BIG;


--미션
--hello_soldesk procedure
--var_hello 변수에 여러분 과정명을 저장해서 출력

create or replace procedure hello_soldesk
is
 var_hello varchar2(1000):='(빅데이터분석) 최적화된 도구(R/파이썬)를 활용한
 애널리스트 양성과정';
 begin
 dbms_output.put_line(var_hello);
 end;
 /
 exec hello_soldesk;
show error; --에러의 내용 확인
////////////////


--데이터를 입력받아서 처리
/
create or replace procedure hello_big
(p_message in varchar2)
is 
begin
  dbms_output.put_line(p_message);
end;
/

--실행
exec hello_big; --실행에러
exec hello_big('hi chulsu');

-- 위 내용에 기본값 할당
 /
create or replace procedure hello_big
(p_message in varchar2:='no message')
is 
begin
  dbms_output.put_line(p_message);
end;
/
exec hello_big;
exec hello_big('messate send');
/

--프로시저의 정보확인
select*from user_objects;
select*from user_objects
where object_name='HELLO_BIG';
//////////////

--Insert 문을 프로시저로 작성
-- departments
desc departments;
select*from departments;

--insert 문 작성
insert into departments values(280,'hong',null,1700);
insert into departments values(300,'hong',null,100);
select*from departments;

-- step3 프로시저 작성(외부에서 데이터 전달하는 매개값 전달
create or replace procedure add_depart
(
p_department varchar2, mgr_id number,loc_id number
)
is
--선언
begin

--실행
insert into departments values(departments_seq.nextval,
p_department,mgr_id,loc_id);
end;
/
--실행
exec add_depart('seoul',200,1700); -- '1700'이외의 다른 숫자를
-- 넣으려면 접속 테이블 location 클릭
-- 데이터 클릭후 LOCATION_ID 번호중에 하나를 넣어야 함
select*from departments;

--시퀀스 확인
desc user_objects;
select*from user_objects;

select object_name, object_type, created
from user_objects
where object_type='SEQUENCE';

select departments_seq.currval from dual; --nextval이 한 번 시행되어야만 실행가능
select departments_seq.nextval from dual;
select*from departments;

COMMIT;

--3/23 수업
--문법순서
--select
--from
--where
--group by
--having
--order by

--실행순서
--from
--where
--group by
--having
--select
--order by

--join 미션
--professor 테이블에서 교수번호, 교수이름, 입사일,
--자신보다 입사일이 빠른 사람의 인원 수를 출력하세요.
--단, 자신보다 입사일이 빠른 사람 수를 오름차순출력,

select*from professor;
select p1.profno, p1.name, p1.hiredate, count(p2.hiredate
)count
from professor p1 left outer join professor p2
on p1.hiredate > p2.hiredate --자신보다 입사일이
--빠른 날짜를 조건으로 표현
group by p1.profno, p1.name, p1.hiredate
order by 4;  --ansl style

select p1.profno,p1.name,p1.hiredate,count(p2.hiredate)count
from professor p1, professor p2
where p1.hiredate > p2.hiredate(+)
group by p1.profno, p1.name, p1.hiredate
order by 4; --oracle style

--서브쿼리 미션
--emp 테이블에서 부서에서 최소급여를 받는 사람들의
--이름, 급여, 부서번호 정보 출력
select*from emp;
--1. 부서에서 최소급여
select min(sal)
from emp
group by deptno;

--2. 
select ename, sal, deptno
from emp
where sal in(select min(sal)
from emp
group by deptno);

--procedure 미션
--테이블 만들기
create table board2(bid number,
 btitle varchar2(30),
bcontent varchar2(100), bwriter varchar2(20),
bstep number);

desc board2;
select*from board2;

--sequence: board2_seq bid에 적용
--나머지 value값은 prodecure를 이용해 데이터 삽입
--procedure name: brd2_insert

--시퀀스 만들기
create sequence board2_seq;
select board2_seq.nextval from dual;

--procedure 만들기
create or replace procedure brd2_insert
(
title varchar2, content varchar2, writer varchar2,
step number
)
is
begin
  insert into board2 values(board2_seq.nextval
  , title, content, writer, step);
  dbms_output.put_line(board2_seq.currval ||','|| title); 
  --실행하면서 출력해보기
end;
/

exec brd2_insert('big data','빅데이터전망','hong1',2);
exec brd2_insert('com science', '프로그래밍', 'hong2', 3);
exec brd2_insert('python','python 프로그래밍','hong3',4);
--board2의 insert 문장만들기
insert into board2 values(100,'aa','bb','cc',5);
select*from board2;

--반복문
create or replace procedure sumprint
is
 num1 number :=0;
 sum1 number :=0; --합으로 사용
begin
 num1:=num1+1;
 sum1:=sum1+num1;
 dbms_output.put_line(num1 ||','|| sum1);
end;
/
--실행
exec sumprint;

--위 문장을 반복문을 사용하여 
create or replace procedure sumprint
is
 num1 number :=0;
 sum1 number :=0; --합으로 사용
begin 
loop 
num1:=num1+1;
 sum1:=sum1+num1;
 dbms_output.put_line('num :'||num1||',sum:'||sum1);
--빠져나오는 문장
exit when num1=10;
end loop;
end;
/

 
--실행
exec sumprint;

/////////////////////////

--loop문을 적용해서 데이터 5개 동시 입력
-- 시퀀스 적용, big data1, 빅데이터의 전망1, hong1, 2
-- 시퀀스 적용, big data2, 빅데이터의 전망2, hong2, 2
-- 시퀀스 적용, big data3, 빅데이터의 전망3, hong3, 2
-- 시퀀스 적용, big data4, 빅데이터의 전망4, hong4, 2
-- 시퀀스 적용, big data5, 빅데이터의 전망5, hong5, 2

--시퀀스 만들기
create sequence board3_seq;
select board3_seq.nextval from dual;

--procedure 만들기
create or replace procedure sumprint
is
 num1 number :=0;
 sum1 number :=0; --합으로 사용
begin 
loop 
num1:=num1+1;
 sum1:=sum1+num1;
 dbms_output.put_line('num :'||num1||',sum:'||sum1);
--빠져나오는 문장
exit when num1=10;
end loop;
end;
/

 
--실행
set serveroutput on;
exec brd2_inset('big data','빅데이터의 전망','hong',2);


--미션2
--update, bid와, title content를 입력받아 bid조건의 내용을 update
--procedure name: board2_update
--exec board2_update(3, 'python3','sd pythen');
--추가데이터
--1. big data, 빅데이터의 전망, hong1, 2
--2. com science, 프로그래밍, hong2, 3
--3. google, google datacenter, hong3, 5
--4. naver, naver datacenter, hong4, 3
--5. python, language python, hong5, 1

create table board2(bid number,
 btitle varchar2(30),
bcontent varchar2(100), bwriter varchar2(20),
bstep number);

--시퀀스 만들기
create sequence board2_seq;
select board2_seq.nextval from dual;

--procedure 만들기
create or replace procedure board2_update
(
id number
title varchar2
content varchar2
)
is
begin
 update board2 set btitle=title, bcontent=content where bid=id;
 dbms_output.put_line(id||','||title||','||content);
end;
/

set bititle='aaa',bcontent='bbb' where bid=1;
/

--exec board2_update(3, 'python3','sd pythen');
update board2
set btitle='aaa',bcontent='bbb'
where bid=1;
select*from board2;

--실행
exec board2_update(5,'cccc','cccccc~~');
select*from board2;

--procedure 3 마무리
select*from employees;
--hiredate의 연도를 입력받아 select 결과 출력하는 프로시저
--연도만 출력
select extract(year from hire_date) from employees;
--반대로 2003 연도를 조건으로 데이터 출력
select employee_id, last_name, hire_date
from employees
where extract(year from hire_date) = 2001;

--result
--100 king 03/06/17
--115 khoo 03/05/18
--122 kaufling 03/05/01
--137 Ladwig 03/07/14
--141 Rajs 03/10/17
--200 Whalen 03/09/17

--procedure name: yearselect
create or replace procedure yearselect
is
begin

-- into 절
select employee_id,last_name,hire_date
from employees
where extract(year from hire_date)=2003;
end;
/

--step4
--cursor 활용
--id employees.employee_id%TYPE;
--name employees.last_name%TYPE;
--hiredate employees.hire_date%TYPE;
create or replace procedure yearselect
(p_year number) --매개변수전달(year)
is
 emp employees%ROWTYPE;
--커서의 선언
 cursor emp_cur is select employee_id,last_name, hire_date
 from employees
 where extract(year from hire_date)=p_year;

begin 
 --커서 오픈
 OPEN emp_cur;
 --커서에서 데이터 읽기
 fetch emp_cur into emp.employee_id,
 emp.last_name,emp.hire_date;
 dbms_output.put_line(emp.employee_id||':'||
 emp.last_name||':'||emp.hire_date);
 --커서달기
 close emp_cur;
 end;
 /
 
select employee_id,last_name,hire_date

--into 절
into id, name, hiredate
from employees
where extract(year from hire_date)=p_year;
dbms_output.put_line(id||':'||name||':'||hiredate);

end;
/

exec yearselect(2003);


--step5
--반복
--id employees.employee_id%TYPE;
--name employees.last_name%TYPE;
--hiredate employees.hire_date%TYPE;
create or replace procedure yearselect
(p_year number) --매개변수전달(year)
is
 emp employees%ROWTYPE;
--커서의 선언
 cursor emp_cur is select employee_id,last_name, hire_date
 from employees
 where extract(year from hire_date)=p_year;

begin 
 --커서 오픈
 OPEN emp_cur;
 loop
 --커서에서 데이터 읽기
 fetch emp_cur into emp.employee_id,
 emp.last_name,emp.hire_date;
 dbms_output.put_line(emp.employee_id||':'||
 emp.last_name||':'||emp.hire_date);
 --반복문 빠져나오기
 exit when emp_cur%notfound;
 end loop;
 --커서 달기
 close emp_cur;
 end;
 /
 
select employee_id,last_name,hire_date

--into 절
into id, name, hiredate
from employees
where extract(year from hire_date)=p_year;
dbms_output.put_line(id||':'||name||':'||hiredate);

end;
/

exec yearselect(2003);

--미션 student select
--select*from student;
--deptno1 입력하면
--studno,name, deptno1 출력될 수 있도록 프로시저작성
--프로시저이름: stu_select


--trigger 방아쇠 처럼 신호를 받아 동작
create table test01(
no number not null,
name varchar2(10),
a number,
b number);

insert into test01 values(1,'a',10,20);
select count(*) from test01;



create table test02
as
select*from test01 where 1=2;
select*from test02;
delete from test01;
--테이블 모두 데이터 없음

create or replace trigger in_test02
after insert on test01
for each row
declare
begin
insert into test02
values(:new.no, :new.name, :new.a, :new.b);
end;
/

--test01에 데이터 추가
insert into test01 values(3,'k',10,20);
insert into test01 values(5,'j',10,20);

select*from test01;
select*from test02;

--trigger 삭제
drop trigger in_test02;
insert into test01 values(6,'j',10,20);
select*from test01;
select*from test02;

--21/03/24
index
backup
mysql 사용법 약간 문제풀이

--41번 문제 입사일로부터 90일이 지난 후의 사원이름, 입사일, 90일 후의 날,
--급여를 출력하라.

select*from emp;
select ename, hiredate, hiredate+90, to_char(sal, '$999,999')
from emp;

--42번 문제 입사일로부터 6개월이 지난 후의 입사일, 6개월 후의 날짜, 급여를 출력하라.
select hiredate, add_months(hiredate,6), to_char(sal, '$999,999')
from emp;

--43번 문제 입사한 달의 근무일 수를 계산하여 부서번호, 이름, 근무일수를 출력하라.
select deptno, ename, last_day(hiredate)-hiredate"근무일수"
from emp;


--44번 모든 사원의 60일이 지난 후의 MONDAY는 몇 년, 몇 월, 몇 일인가를 구하여
--이름, 입사일, MONDAY를 출력하라.


--45번 입사일로부터 오늘까지의 일수를 구하여 이름, 입사일, 근무일수를 출력하라.
select ename, hiredate, trunc((sysdate-hiredate),0)
from emp;

select*from emp;


--INDEX--
INDEX란 주소록 같은 개념
index 주는 칼럼은 where절에 들어가는 칼럼을 주로 사용
index split현상 데이터를 삽입삭제가 빈번한 곳에는 현상
데이터삭제되면 테이블에서는 없어지지만 index에서는 남아있음 
dml작업이 빈번한 곳에 취약



/
;
select rowid, empno, ename
from emp
where empno=7902;

select*from emp;

--index 생성
--create index 인덱스 이름 on 테이블이름(칼럼이름)
create index emp_idx on emp(empno);

select*from user_indexes
where table_name='EMP';

--index성능테스트
--테이블준비
create table emp10
as
select*from bigemp1 order by dbms_random.value;

create table emp_idx
as
select*from bigemp1 order by dbms_random.value;

select*from emp10 where table_name='EMP10';
select*from emp_idx where table_name='emp_idx';

--index를 emp_idx테이블에 만들기
create index idx_empidx_empno on emp_idx(emp_no);

select index_name,index_type.blevel.leaf_blocks.distinct_kyes.num_rows
from user_indexes
where table_name='emp_idx';

--성능비교
select*from emp10 where emp_no < 10100;

select*from emp_idx where emp_no < 10100;

--emp10 0.196초 -> 0.193초
--EMPIDX 0.179초 ->0.169초

--같은 조건에서 조회 결과

--index rebuild (balancing 확인)
create table inx_test
( no number);

/
begin
for i in 1...10000 loop
insert into inx_test values(i);
end loop;
commit;
end;
/
select count(*) from inx_test;

--index 생성
create index idx_inxtest_no on inx_test(no);

--밸런스 조회
--분석
analyze index idx_inxtest_no validate structure;
--밸런스 조회
select (del_lf_rows_len/lf_rows_len)*100 balance
from index_stats
where name='IDX_INXTEST_NO';

--부분삭제
DELETE FROM INX_TEST
WHERE NO BETWEEN 1 AND 4000;

SELECT COUNT(*) FROM INX_TEST;

--밸런스 조회
select (del_lf_rows_len/lf_rows_len)*100 balance
from index_stats
where name='IDX_INXTEST_NO';

--REBUILD하기
ALTER INDEX IDX_INXTEST_NO REBUILD;

--다시 분석 
analyze index idx_inxtest_no validate structure;

--다시 밸런스 조회
select (del_lf_rows_len/lf_rows_len)*100 balance
from index_stats
where name='IDX_INXTEST_NO';

---------------------------------------------------------------------------






