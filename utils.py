import re
import time

def execution_time(process):
    """ Calculate the execution time of a process """
    start_time = time.time()
    process()
    end_time = time.time()
    return (end_time - start_time)