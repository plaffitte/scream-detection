def text_to_latex(results,class_list,path):
    no_classes = len(class_list)
    text = "\\begin{table}[t]\n"
    text = text + "\\begin{centering}\n"
    text = text + "\\begin{tabular}{"
    for i in range(no_classes-1):
        text = text + "c"
    text = text + "}\n"
    text = text + "\\hline\n"
    for i in range(no_classes):
        text  = text + "& $" + class_list[i] + "$ "
    text = text + "\\tabularnewline\n"
    text = text + "\\hline\n"
    for i in range(no_classes):
        text = text + "$" + class_list[i] + "$ "
        for j in range(no_classes):
            if i==j:
                text = text + "& \\textbf{ " + str(results[i,j]) + "}"
            else:
                text = text + "& " + str(results[i,j])

        text = text + " \\tabularnewline\n"
        text = text + "\\hline\n"

    text = text + "\\par\\end{centering}\n" + "\\caption{"
    for i in range(no_classes-1):
        text = text + class_list[i] + " vs "
    text = text + class_list[no_classes-1]
    text = text + "}\n" + "\\end{table} \n\\end{tabular}"
    logfile = path + "/results.tex"
    log = open(logfile, 'w')
    log.write(text)
    log.close()
