Test rose suite for creating a lagged ensemble with aligned validation periods. 

For example, initialization in s1963 would include:

* s1963 - year 2-9 mean - period from Dec 1964 to Mar 1972
* s1962 - year 3-9 mean - period from Dec 1964 to Mar 1971
* s1961 - year 4-9 mean - period from Dec 1964 to Mar 1970
* s1960 - year 5-9 mean - period from Dec 1964 to Mar 1969

Increases the ensemble size (quadruples the members) at the expense of relying on persistence.

Update!

To have the same overlapping period for all files and considering s1960 as the first initialization year. This would include

* s1963 (init - 0) - year 2-6 average - period from DJFM 1964/1965 to DJFM 1968/1969
* s1962 (init - 1) - year 3-7 average - period from  DJFM 1964/1965 to DJFM 1968/1969
* s1961 (init - 2) - year 4-8 average - period from  DJFM 1964/1965 to DJFM 1968/1969
* s1960 (init - 3) - year 5-9 average - period from  DJFM 1964/1965 to DJFM 1968/1969

All of these cover the same overlapping period.

In paper 1, I plan to consider alternative lag schemes for the following forecast ranges: years 2-5, years 2-3, year 2, and year 1.

A worked example of years 2-5 lagging (with a four year lag) would include:

* s1963 (init - 0) - year 2-5 average - period from DJFM 64/65 to DJFM 67/68
* s1962 (init - 1) - year 3-6 average - period from DJFM 64/65 to DJFM 67/68
* s1961 (init - 2) - year 4-7 average - period from DJFM 64/65 to DJFM 67/68
* s1960 (init - 3) - year 5-8 average - period from DJFM 64/65 to DJFM 67/68

A worked example of years 2-3 lagging (with a four year lag) would include:

* s1963 (init - 0) - year 2-3 average - period from DJFM 64/65 to DJFM 65/66
* s1962 (init - 1) - year 3-4 average - period from DJFM 64/65 to DJFM 65/66
* s1961 (init - 2) - year 4-5 average - period from DJFM 64/65 to DJFM 65/66
* s1960 (init - 3) - year 5-6 average - period from DJFM 64/65 to DJFM 65/66

A worked example of second year lagging (with a four year lag) would include:

* s1963 (init - 0) - year 2 average - period from DJFM 64/65
* s1962 (init - 1) - year 3 average - period from DJFM 64/65
* s1961 (init - 2) - year 4 average - period from DJFM 64/65
* s1960 (init - 3) - year 5 average - period from DJFM 64/65

A worked example of first year lagging (with a four year lag, assuming nov. initialisation) would include:

* s1963 (init - 0) - year 1 average - period from DJFM 63/64
* s1962 (init - 1) - year 2 average - period from DJFM 63/64
* s1961 (init - 2) - year 3 average - period from DJFM 63/64
* s1960 (init - 3) - year 4 average - period from DJFM 63/64