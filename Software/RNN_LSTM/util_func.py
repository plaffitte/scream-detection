import sys
from datetime import datetime

def parse_classes(class_str):
    class_list = []
    start = 1
    i=0
    while not class_str[i+1] is '}':
        while class_str[i+1] not in {',', '}'}:
            i += 1
        end = i+1
        class_list.append(class_str[start:end])
        if not class_str[i+1] is '}':
            i += 1
        start = i + 1
    return class_list

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--", "").replace("-", "_")
        args[key] = arg_elements[2*i+1]
    return args

def log(string):
    sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')
    print('[' + str(datetime.now()) + '] ' + str(string) + '\n')
