# modules we'll use
# https://strftime.org/
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

df = pd.DataFrame({
    'date': ['3/2/07', '3/22/07', '1/17/07'], 
    'date2': ['3-2-2007', '22-3-2007', '17-1-2007'],
    'multi': ['3/2/07', '3/22/07', '17-1-2007']
})
df['date_parsed'] = pd.to_datetime(df['date'], format="%m/%d/%y")
df['date2_parsed'] = pd.to_datetime(df['date2'], format="%d-%m-%Y")
df['multi_parsed'] = pd.to_datetime(df['multi'], infer_datetime_format=True)


days_df = df['date_parsed'].dt.day
monthes_df = df['date_parsed'].dt.month
year_df = df['date_parsed'].dt.year
a = df['date_parsed'].dt.day_of_week
b = df['date_parsed'].dt.day_of_year
x = df['date_parsed'].dt.weekday
c = df['date_parsed'].dt.minute
c = df['date_parsed'].dt.second

print(df)
print(days_df)
print(monthes_df)
print(year_df)




'''
Code	Example	Description
%a	Sun	Weekday as locale’s abbreviated name.
%A	Sunday	Weekday as locale’s full name.
%w	0	Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.
%d	08	Day of the month as a zero-padded decimal number.
%-d	8	Day of the month as a decimal number. (Platform specific)
%b	Sep	Month as locale’s abbreviated name.
%B	September	Month as locale’s full name.
%m	09	Month as a zero-padded decimal number.
%-m	9	Month as a decimal number. (Platform specific)
%y	13	Year without century as a zero-padded decimal number.
%Y	2013	Year with century as a decimal number.
%H	07	Hour (24-hour clock) as a zero-padded decimal number.
%-H	7	Hour (24-hour clock) as a decimal number. (Platform specific)
%I	07	Hour (12-hour clock) as a zero-padded decimal number.
%-I	7	Hour (12-hour clock) as a decimal number. (Platform specific)
%p	AM	Locale’s equivalent of either AM or PM.
%M	06	Minute as a zero-padded decimal number.
%-M	6	Minute as a decimal number. (Platform specific)
%S	05	Second as a zero-padded decimal number.
%-S	5	Second as a decimal number. (Platform specific)
%f	000000	Microsecond as a decimal number, zero-padded to 6 digits.
%z	+0000	UTC offset in the form ±HHMM[SS[.ffffff]] (empty string if the object is naive).
%Z	UTC	Time zone name (empty string if the object is naive).
%j	251	Day of the year as a zero-padded decimal number.
%-j	251	Day of the year as a decimal number. (Platform specific)
%U	36	Week number of the year (Sunday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Sunday are considered to be in week 0.
%-U	36	Week number of the year (Sunday as the first day of the week) as a decimal number. All days in a new year preceding the first Sunday are considered to be in week 0. (Platform specific)
%W	35	Week number of the year (Monday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Monday are considered to be in week 0.
%-W	35	Week number of the year (Monday as the first day of the week) as a decimal number. All days in a new year preceding the first Monday are considered to be in week 0. (Platform specific)
%c	Sun Sep 8 07:06:05 2013	Locale’s appropriate date and time representation.
%x	09/08/13	Locale’s appropriate date representation.
%X	07:06:05	Locale’s appropriate time representation.
%%	%	A literal '%' character.
'''