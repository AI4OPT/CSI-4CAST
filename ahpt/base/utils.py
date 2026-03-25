import datetime


TIME_FORMAT = "%Y%m%d_%H%M%S"


def get_current_time():
    return datetime.datetime.now().strftime(TIME_FORMAT)
