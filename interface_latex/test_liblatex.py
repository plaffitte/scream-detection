import numpy as np
import sys


sys.path.insert(0, "/media/DATA/home/sodoyer/Code_Python/Pierre_DNN/interface_latex")

from liblatex import *


a = np.random.randn(2,2)
print a

Docpath = "/media/DATA/home/sodoyer/Code_Python/Pierre_DNN/test_doc_tex"
TabPath = "/media/DATA/home/sodoyer/Code_Python/Pierre_DNN/test_doc_tex"
Docname = "document"
TabName = "tableau"
texfile = DocTabTex(a,['label_1','lable_2'],"Tab01","legende",Docpath,TabPath,Docname,TabName,option='t',commentaire="%test ommanataire")
texfile.creatTex();
texfile.creatpdf();
