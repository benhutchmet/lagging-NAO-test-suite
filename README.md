Test rose suite for creating a lagged ensemble with aligned validation periods. 

For example, initialization in s1963 would include:

* s1963 - year 2-9 mean - period from Dec 1964 to Mar 1972
* s1962 - year 3-9 mean - period from Dec 1964 to Mar 1971
* s1961 - year 4-9 mean - period from Dec 1964 to Mar 1970
* s1960 - year 5-9 mean - period from Dec 1964 to Mar 1969

Increases the ensemble size (quadruples the members) at the expense of relying on persistence.

Update!

To have the same overlapping period for all files and considering s1960 as the first initialization year. This would include

* s1963 (init - 0) - year 2-6 average - period from DJFM 1964/1965 to DJFM 1969/1970
* s1962 (init - 1) - year 3-7 average - period from  DJFM 1964/1965 to DJFM 1969/1970
* s1961 (init - 2) - year 4-8 average - period from  DJFM 1964/1965 to DJFM 1969/1970
* s1960 (init - 3) - year 5-9 average - period from  DJFM 1964/1965 to DJFM 1969/1970

All of these cover the same overlapping period.