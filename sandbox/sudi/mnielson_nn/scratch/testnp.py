import numpy as np

class TestsClass:
    def __init__ (self, w=1, b=2):
        self.w = w
        self.b = b

    def PrintObj(self):
        print "Data:",self.w,self.b



Obj = TestsClass()
Obj.PrintObj()
# ---------------#
# list comprehension
x = [indx for indx in range(0,49,3) if (indx%2==0)]
print x

y = [v*2 for v in x]
print y
z = (0,'dad'),(1,'mum'), (2,'son'),(2,'sonny')
print z, len(z), type(z)

a = [1,2,3]
aa,bb,cc = a
print aa,bb,cc

if (5 in a):
    print "list a contains 5\n"
# dictionaries
dct = {z[0]:'home',z[1]:'work',z[2]:'school'}
print dct

aa,bb,cc = dct
print "as individual",aa,bb,cc

#del dct['sudi']
print dct
# sets
seta = {'app','app','bad'}
seta.add('bada')
print seta,"size=",len(seta)

for indx,x in enumerate(dct):
    print indx,x,'@',dct[x]

for name,val in dct.items():
    print ('name=%-20s @=%-s' %(name,val))
# =========
# numpy
x = np.array([
[00, 01],
[10, 11]
])
print x.shape
print x[0,0], x[0,1], x[1,0],x[1,1]

z = np.random.random((4,4))
print z
