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
        text  = text + '& $' + '\\texbf{' + class_list[i] + '}' + '$ '
    text = text + 'Total error' + '\\' + '\n'
    text = text + '\\hline\n'
    for i in range(no_classes):
        text = text + '$' + '\\texbf{' + class_list[i] + '}' + '$ '
        for j in range(no_classes):
            if i==j:
                text = text + '&' + '\\cellcolor{lightgray}' + '\\textbf{ ' + str(results[i,j]) + '}'
            else:
                text = text + '& ' + str(results[i,j])

        text = text + ' \\' + '\n'
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
