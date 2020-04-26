from sklearn.ensemble import RandomForestClassifier

traindata=[]
testdata={}
class Domain:
    def __init__(self,_name,_label,_length,_countnum):
        self.name=_name
        self.label=_label
        self.length=_length
        self.countnum=_countnum

    def returndata(self):
        return [self.length,self.countnum]

    def returnlabel(self):
        if self.label=="dga":
            return 1
        else:
            return 0

def init_train(filename):
    with open(filename) as f:
        for line in f:
            line=line.strip()
            if line.startswith("#") or line=="":
                continue
            tokens=line.split(",")
            name=tokens[0]
            label=tokens[1]
            length=len(name)
            countnum=0
            for i in name:
                if i.isdigit():
                    countnum+=1
            traindata.append(Domain(name,label,length,countnum))

def init_test(filename):
    with open(filename) as m:
        for line in m:
            length=len(line)
            countnum=0
            for j in line:
                if j.isdigit():
                    countnum+=1
            testdata[line]=[length,countnum]

def main():
    init_train("train.txt")
    init_test("test.txt")
    trainmatrix=[]
    labellist=[]
    for item in traindata:
        trainmatrix.append(item.returndata())
        labellist.append(item.returnlabel())

    clf=RandomForestClassifier(random_state=0)
    clf.fit(trainmatrix,labellist)
    doc=open("result.txt","w")
    for item2 in testdata.items():
        temp=clf.predict([item2[1]])
        if temp==1:
            label='dga'
        else:
            label='notdga'
        print(item2[0][:-1],label,sep=',',file=doc)
    doc.close()

if __name__=='__main__':
    main()
