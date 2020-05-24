from pyvi import ViTokenizer, ViPosTagger,ViUtils

print(ViTokenizer.tokenize(u"Trung Quốc tố Mỹ vu khống WHO để trốn trách nhiệm với COVID-19"))

print(ViPosTagger.postagging(ViTokenizer.tokenize(u"Trung Quốc tố Mỹ vu khống WHO để trốn trách nhiệm với COVID-19")))

#from pyvi import
print(ViUtils.remove_accents(u"Trường đại học bách khoa hà nội"))

#from pyvi import ViUtils
print(ViUtils.add_accents(u'thu tuong yeu cau lam ro trach nhiem'))