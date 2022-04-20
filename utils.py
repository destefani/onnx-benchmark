import re
import time

def execution_time(process):
    """ Calculates the execution time of a function """
    start_time = time.time()
    (lambda: process)()
    end_time = time.time()
    return (end_time - start_time)