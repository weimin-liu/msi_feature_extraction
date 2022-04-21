"""
this module includes all the functions related to time
"""
import time
from datetime import datetime as dt


def since_epoch(date):
    """
    return the time span in seconds since epoch
    """
    return time.mktime(date.timetuple())


def from_year_fraction(date: float):
    """
    this function takes a date in float as input and return datetime as output

    Parameter:
    --------
        date: float such as 2000.02

    Return:
    --------
        the corresponding datetime object
    """
    if type(date).__name__ != 'float':
        raise TypeError('date must be a float')
    year = int(date)
    fraction = date - year
    start_of_this_year = dt(year=year, month=1, day=1)
    start_of_next_year = dt(year=year + 1, month=1, day=1)
    year_duration = since_epoch(start_of_next_year) - since_epoch(start_of_this_year)
    year_elapsed = fraction * year_duration
    return dt.fromtimestamp(year_elapsed + since_epoch(start_of_this_year))


def to_year_fraction(date):
    """
    this function takes datetime object as input and return float date as output

    Parameter:
    --------
        date: a datetime object

    Return:
    --------
        the corresponding float of the datetime object
    """
    year = date.year
    start_of_this_year = dt(year=year, month=1, day=1)
    start_of_next_year = dt(year=year + 1, month=1, day=1)
    year_elapsed = since_epoch(date) - since_epoch(start_of_this_year)
    year_duration = since_epoch(start_of_next_year) - since_epoch(start_of_this_year)
    fraction = year_elapsed / year_duration
    return date.year + fraction
