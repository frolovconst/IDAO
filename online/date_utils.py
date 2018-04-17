from datetime import timedelta, date
import pandas as pd

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_holidays_set():
    holidays = []
    #2015
    for i in range(1, 10):
        holidays += [pd.to_datetime("201501{}".format(i), format='%Y%m%d')]
    holidays += [pd.to_datetime("20150223", format='%Y%m%d')]
    holidays += [pd.to_datetime("20150309", format='%Y%m%d')]
    holidays += [pd.to_datetime("20150501", format='%Y%m%d')]
    holidays += [pd.to_datetime("20150504", format='%Y%m%d')]
    holidays += [pd.to_datetime("20150511", format='%Y%m%d')]
    holidays += [pd.to_datetime("20150612", format='%Y%m%d')]
    holidays += [pd.to_datetime("20151104", format='%Y%m%d')]

    #2016
    for i in range(1, 10):
        holidays += [pd.to_datetime("201601{}".format(i), format='%Y%m%d')]
    holidays += [pd.to_datetime("20160222", format='%Y%m%d')]
    holidays += [pd.to_datetime("20160223", format='%Y%m%d')]
    holidays += [pd.to_datetime("20160307", format='%Y%m%d')]
    holidays += [pd.to_datetime("20160308", format='%Y%m%d')]
    holidays += [pd.to_datetime("20160502", format='%Y%m%d')]
    holidays += [pd.to_datetime("20160503", format='%Y%m%d')]
    holidays += [pd.to_datetime("20160509", format='%Y%m%d')]
    holidays += [pd.to_datetime("20160613", format='%Y%m%d')]
    holidays += [pd.to_datetime("20161104", format='%Y%m%d')]

    #2017
    for i in range(1, 9):
        holidays += [pd.to_datetime("201701{}".format(i), format='%Y%m%d')]
    holidays += [pd.to_datetime("20170223", format='%Y%m%d')]
    holidays += [pd.to_datetime("20170224", format='%Y%m%d')]
    holidays += [pd.to_datetime("20170308", format='%Y%m%d')]
    holidays += [pd.to_datetime("20170501", format='%Y%m%d')]
    holidays += [pd.to_datetime("20170508", format='%Y%m%d')]
    holidays += [pd.to_datetime("20170509", format='%Y%m%d')]
    holidays += [pd.to_datetime("20170612", format='%Y%m%d')]
    holidays = set(holidays)
    return holidays
    
def get_weeknds_set():
    start_date = date(2014, 12, 1)
    end_date = date(2017, 10, 1)
    weeknds = get_holidays_set()
    for single_date in daterange(start_date, end_date):
        if (single_date.weekday() == 5 or single_date.weekday() == 6):
            weeknds.add(pd.Timestamp(single_date))
    return weeknds
    
def get_days_till_weeknd(weeknds):
    result = {}
    
    till_weeknd = 0
    start_date = date(2015, 1, 1)
    end_date = date(2017, 10, 1)
    for single_date in daterange(start_date, end_date):
        time_stamp = pd.Timestamp(single_date)
        if till_weeknd < 0:
            delta = 0            
            while time_stamp + pd.Timedelta(days=delta) not in weeknds:
                delta += 1
            till_weeknd = delta
        result[time_stamp] = till_weeknd
        till_weeknd -= 1
    return result

def get_days_from_weeknd(weeknds):
    result = {}
    
    from_weeknd = 0
    start_date = date(2015, 1, 1)
    end_date = date(2017, 10, 1)
    for single_date in daterange(start_date, end_date):
        time_stamp = pd.Timestamp(single_date)
        if time_stamp in weeknds:
            from_weeknd = 0
        result[time_stamp] = from_weeknd
        from_weeknd += 1
    return result

def get_number_of_weeknds(weeknds):
    result = {}
    
    start_date = date(2014, 12, 22)
    end_date = date(2017, 10, 1)
    num_of_weeknds = 0
    cur_days = []
    for single_date in daterange(start_date, end_date):
        time_stamp = pd.Timestamp(single_date)
        if time_stamp in weeknds:
            num_of_weeknds += 1
        cur_days += [time_stamp]
        
        if single_date.weekday() == 6:
            for cday in cur_days:
                result[cday] = num_of_weeknds
            num_of_weeknds = 0
            cur_days = []

    return result