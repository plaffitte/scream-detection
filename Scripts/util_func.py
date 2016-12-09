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
    i += 1
    if i < len(class_str) - 1:
        if class_str[i+1] is '+':
            i += 2
            start = i + 1
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

def text_to_latex(results,class_list,path):
    no_classes = len(class_list)
    text = '\\begin{table}[h*]\n'
    text = text + '\\centering\n'
    text = text + '\\begin{tabular}{'
    for i in range(no_classes+2):
        text = text + '|c'
    text = text + '}\n'
    text = text + '\\hline\n'
    for i in range(no_classes):
        text  = text + '&' + '\\textbf{' + class_list[i] + '}'
    text = text + '&' + 'Total error' + '\\' + '\\' + '\n'
    text = text + '\\hline\n'
    for i in range(no_classes):
        text = text + '\\textbf{' + class_list[i] + '}'
        for j in range(no_classes):
            if i==j:
                text = text + '&' + '\\cellcolor{lightgray}' + '\\textbf{ ' + str(results[i,j]) + '}'
            else:
                text = text + '& ' + str(results[i,j])

        text = text + ' \\' + '\\' + '\n'
        text = text + '\\hline\n'

    text = text + '\\end{tabular}'
    text = text + '\\caption{'
    for i in range(no_classes-1):
        text = text + class_list[i] + ' vs '
    text = text + class_list[no_classes-1]
    text = text + '}\n' + '\\end{table} \n'
    logfile = path + '/results.tex'
    log = open(logfile, 'w')
    log.write(text)
    log.close()
