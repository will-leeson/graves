import json, numpy as np, borda.count
from scipy.stats import spearmanr

svRes = json.load(open("../../data/SV-CompResults.json"))

choices = ["CPASEQ", "twoLS", "ESBMCIncr","Depthk","UTaipan","Symbiotic","ESBMCKind","UKojak","CBMC","UAutomizer"]

scores = np.array([[0]*10,[0]*10,[0]*10,[0]*10, [0]*10])
for key in svRes:
    problem = int(key.split("|||")[1])
    arr = np.array(svRes[key])
    idx = 0
    for score in arr[:,1].argsort():
        scores[0][score]+=idx
        scores[problem+1][score]+=idx
        idx+=1

staticRanking = scores.argsort()

runningCorr = 0
for key in svRes:
    problem = int(key.split("|||")[1])
    arr = np.array(svRes[key])

    x,_ = spearmanr(arr[:,1], staticRanking[0])
    runningCorr+=x


print(runningCorr)
print(runningCorr/len(svRes))