from matplotlib import pyplot as plt
import json, glob
import numpy as np

labels = json.load(open("../../data/subsetTestFiles.json"))
cstFiles = glob.glob("cst_predicts/*.json")

verifiers = {0:"cpaseq", 1:"2ls", 2:"esbmcincr", 3:"depthk", 4:"utaipan",5:"symbiotic", 6:"esbmckind", 7:'ukojak',8:'cbmc',9:'uautomizer'}
verifiers2Num = {"cpaseq":0, "2ls":1, "esbmcincr":2,"depthk":3,"utaipan":4,"symbiotic":5, "esbmckind":6,'ukojak':7,'cbmc':8,'uautomizer':9}
topKTimes = []
topKVals = [1,8,5,6]
overallTimes = [[] for _ in range(10)]
for problem in labels:
    scores = np.array([x[0] for x in labels[problem]])
    times = np.array([(x[0]-x[1])*1000 for x in labels[problem]])
    
    for i, (score, time) in enumerate(zip(scores, times)):
        if score > 0:
            overallTimes[i].append(time)
    
    minVal = 1000
    for i in topKVals:
        if scores[i] >0 and times[i]< minVal:
            minVal = times[i]
    
    if minVal<1000:
        topKTimes.append(minVal)

for i in range(len(overallTimes)):
    overallTimes[i].sort()

gravesFiles = glob.glob("graves/*.json")
gravesTimes = json.load(open("graph_builder_times.json"))
gravesTimes = {y.split("/")[-1]+".json|||"+str(i):gravesTimes[y] for y in gravesTimes for i in range(4)}

gravesScores = json.load(open(gravesFiles[0]))
gravesOverallTimes = []
gravesTop4OverallTimes = []
for problem in labels:
    res, time = gravesScores[problem]
    scores = [x[0] for x in labels[problem]]
    times = [(x[0]-x[1])*1000 for x in labels[problem]]
    if scores[np.argmax(res)] >0 and times[np.argmax(res)]+time+gravesTimes[problem] <900 :
        gravesOverallTimes.append(times[np.argmax(res)]+time+gravesTimes[problem])
    
    minVal = 1000
    for i in np.argsort(times)[:4]:
        if scores[i] >0 and times[i]< minVal:
            minVal = times[i]
    gravesTop4OverallTimes.append(minVal+time+gravesTimes[problem])

cstScores = json.load(open(cstFiles[0]))
cstOverallTimes = []
cstTop4OverallTimes = []
for problem in labels:
    res, time = cstScores[problem] 
    scores = [x[0] for x in labels[problem]]
    times = [(x[0]-x[1])*1000 for x in labels[problem]]
    verifiersScores = [0]*10
    for i in res:
        verifiersScores[verifiers2Num[i[0]]]=i[1]
    if scores[np.argmax(verifiersScores)] >0 and times[np.argmax(verifiersScores)]+time <900 :
        cstOverallTimes.append(times[np.argmax(verifiersScores)]+time)
    
    minVal = 1000
    for i in np.argsort(time)[:4]:
        if scores[i] >0 and times[i]< minVal:
            minVal = times[i]
    cstTop4OverallTimes.append(minVal+time)


threshholds = [10**(x*0.25) for x in range(-4,12)]

y = [np.sum(np.array(gravesOverallTimes)<j) for j in threshholds]
plt.plot(y,threshholds, label="Graves", marker='o')

# y = [np.sum(np.array(cstOverallTimes)<j) for j in threshholds]
# plt.plot(y, threshholds, label="CST", marker='o')

markers = ['v','^','s','p','+','x','D','*','1','|']
verifiers = {0:"CPA-Seq", 1:"2LS", 2:"ESBMC-incr", 3:"DepthK", 4:"UTaipan",5:"Symbiotic", 6:"ESBMC-kind", 7:'UKojak',8:'CBMC',9:'UAutomizer'}
for i in range(len(overallTimes)):
    y = [np.sum(np.array(overallTimes[i])<j) for j in threshholds]
    plt.plot(y, threshholds, label=verifiers[i], marker=markers[i])

plt.grid()
plt.xlabel("Problems Solved")
plt.ylabel("Time (s)")
plt.yscale('log')
plt.legend(loc=2, prop={'size': 6})
plt.savefig("cactusTop1.png", dpi=300)
plt.clf()

# y = [np.sum(np.array(gravesTop4OverallTimes)<j) for j in threshholds]
# plt.plot(threshholds,y, label="Graves Top-4", marker='o')

# y = [np.sum(np.array(cstTop4OverallTimes)<j) for j in threshholds]
# plt.plot(threshholds,y, label="CST Top-4", marker='o')

# y = [np.sum(np.array(topKTimes)<j) for j in threshholds]
# plt.plot(threshholds,y, label="ISS Top-4", marker='o')
# plt.grid()
# plt.ylabel("Problems Solved")
# plt.xlabel("Time Budget (s)")
# plt.xscale('log')
# plt.legend()
# plt.savefig("cactusTop4.png", dpi=300)