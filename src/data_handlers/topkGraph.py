from matplotlib import pyplot as plt
import csv, json, glob
import numpy as np

labels = json.load(open("../../data/subsetTestFiles.json"))
trainLabels = json.load(open("../../data/subsetTrainFiles.json"))

gravesFiles = glob.glob("graves/*.json")
cstFiles = glob.glob("cst_predicts/*.json")

verifiers = {"cpaseq":0, "2ls":1, "esbmcincr":2,"depthk":3,"utaipan":4,"symbiotic":5, "esbmckind":6,'ukojak':7,'cbmc':8,'uautomizer':9}

cstTopKAll = []
for file in cstFiles:
    CSTChoices = np.array([0]*10)
    score = json.load(open(cstFiles[0]))
    spearman = []
    for problem in labels:
        res, time = score[problem] 
        verifiersScores = [0]*10
        for i in res:
            verifiersScores[verifiers[i[0]]]=i[1]
        scores = np.argsort([-x[1] for x in labels[problem]])
        CSTChoices[np.argwhere(np.argsort(verifiersScores) == scores[0])[0][0]:]+=1
    
    CSTChoices = 1-(CSTChoices/CSTChoices[-1])
    cstTopKAll.append(CSTChoices)

cstTopK = np.mean(cstTopKAll, axis=0)[:9]
cstTopKSTD = np.std(cstTopKAll, axis=0)[:9]  

gravesTopKAll = []
for file in gravesFiles:
    score = json.load(open(file))
    topkVal = np.array([0]*10)
    for problem in labels:
        if len(score[problem]) == 2:
            res, time = score[problem]
        else:
            res = score[problem]
        scores = np.argsort([-x[1] for x in labels[problem]])
        res = np.argsort([-x for x in res[0]])
        topkVal[np.argwhere(res == scores[0])[0][0]:]+=1
    topkVal = topkVal/topkVal[-1]
    topkVal = 1-topkVal
    gravesTopKAll.append(topkVal)

gravesTopK = np.mean(gravesTopKAll, axis=0)[:9]
gravesTopKSTD = np.std(gravesTopKAll, axis=0)[:9]

randomChoices = np.array([[0]*10]*10)

choice = list(range(10))

for i in range(10):
    for problem in labels:
        np.random.shuffle(choice)
        scores = np.argsort([x[1] for x in labels[problem]])
        randomChoices[i][np.argwhere(choice == scores[0])[0][0]] +=1

randomTopK = np.mean(randomChoices, axis = 0)/sum(randomChoices[0])
randomTopK = np.array([1-np.sum(randomTopK[:x+1])for x in range(9)])
randomTopKSTD = (np.std(randomChoices, axis = 0)/np.sum(randomChoices[0]))[:9]

ISSChoices = np.array([0]*10)
trueScores = np.array([0]*10)

for problem in trainLabels:
    scores = np.argsort([x[1] for x in trainLabels[problem]])
    for i in scores:
        trueScores[scores[i]]+=i

choice = np.argsort(-trueScores)

for problem in labels:
    scores = np.argsort([-x[1] for x in labels[problem]])
    ISSChoices[np.argwhere(choice == scores[0])[0][0]] +=1

ISSTopK = ISSChoices/sum(ISSChoices)
ISSTopK = np.array([1-np.sum(ISSTopK[:x+1])for x in range(9)])

plt.plot(range(1,10), gravesTopK, marker='o', label="Graves")
plt.fill_between(range(1,10), gravesTopK-gravesTopKSTD, gravesTopK+gravesTopKSTD, alpha=0.3)

plt.plot(range(1,10), cstTopK[:9], marker='v', label="CST")
plt.fill_between(range(1,10), cstTopK-cstTopKSTD, cstTopK+cstTopKSTD, alpha=0.3)

plt.plot(range(1,10), ISSTopK[:9], marker='s', label="ISS")
plt.fill_between(range(1,10), ISSTopK-0, ISSTopK+0, alpha=0.3)

plt.plot(range(1,10), randomTopK[:9], marker='D', label="Random")
plt.fill_between(range(1,10), randomTopK-randomTopKSTD, randomTopK+randomTopKSTD, alpha=0.3)

plt.grid(True)
plt.xlabel("K-Value")
plt.ylabel("Error")
plt.title("Top-K Error")
plt.legend()
plt.savefig("topk.png", format="png", dpi=300)