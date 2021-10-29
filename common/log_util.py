import sys
import os
import io
import re
import socket
import time
from typing import Tuple, List, Callable, Dict

import pandas as pd
import inspect
import html

def get_hostname():
    """Return host name"""
    return socket.gethostname()

def get_host_ip():
    """ 查询本机ip地址 """
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        ip=s.getsockname()[0]
    finally:
        s.close()

    return ip

def get_timestr():
    """Return time str. format: %Y-%m-%d_%H:%M:%S"""
    timestr = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    return timestr


def get_python_version():
    return '|'.join(sys.version.split('\n'))

def get_cuda_version():
    cuda_versions = re.findall('/(cuda.*?)/', os.environ["PATH"])
    if len(cuda_versions) == 0:
        return ''
    return cuda_versions[-1]



def dataframe2TSVstr(df: pd.DataFrame, float_format: str='%.4f') -> str:
    """ Convert pd.DataFrame data to TSV-style multi-line string
    """
    output = io.StringIO(newline='\n')
    # output.write('First line.\n')
    df.to_csv(output, sep='\t', float_format=float_format)

    # Retrieve file contents
    contents = output.getvalue()
    # Close object and discard memory buffer
    output.close()
    return contents


def func2str(fun: Callable, replace_escape_char=True) -> str:
    """
    Hash functions

    :param fun: function to hash
    :return: hash of the function
    """
    try:
        h = inspect.getsource(fun)
        if replace_escape_char:
            # h = h.replace('\\n', '\n').replace(r"\'", r"'").replace(r'\"', r'"')
            h = html.unescape(h)
    except IOError:
        h = "nocode"
    return h


def log_detail(
    file_name: str,
    message: str='',
    timestamp: bool=False,
    print_detail: bool=False,
    terminal=sys.stdout
):
    """ Write detailed log. If the log file exists, append output to the file.
    """
    message += '\n'
    if timestamp:
        timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        message = '[%s  @PID: %d] ' % (timestr, os.getpid()) + message
    # print to terminal
    if print_detail:
        terminal.write(message)
        terminal.flush()
    # print to log file
    with open(file_name, mode='a') as f:
        f.write(message)
        f.flush()



def log_summary(file_name: str, report_dict: Dict):
    """ Save experiment summary, include time and result.
    Summary filename formart: ./log/<Year-Month-Day>_<name_str>.log
    If the log file exists, append output to the file.

    Args:
        report_dict: dict, key is a str, value must be convertible into a string
    """
    timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    with open(file_name, mode='a') as f:
        # write head
        f.write('\n' + '====' * 20 + '\n')
        f.write('Time: %s    PID:%d\n' % (timestr, os.getpid()))
        # write key - value
        for key in report_dict:
            f.write('%s:' % key)
            value = report_dict[key]
            f.write(str(value))
            f.write('\n')
        f.write('\n')



if __name__ == '__main__':
    print(get_host_ip())
    print(get_hostname())
    print(get_timestr())
    print(get_python_version())
    print(get_cuda_version())
