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
optimalTimes = []
for problem in labels:
    scores = np.array([x[0] for x in labels[problem]])
    times = np.array([(x[0]-x[1])*1000 for x in labels[problem]])

    if sum(scores>0)>0:
        optimalTimes.append(times[scores>0].min())
    
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
    if i == 0:
        print("Verification Time")
        print("Min:", np.min(overallTimes[i]))
        print("Max:", np.max(overallTimes[i]))
        print("Mean:", np.mean(overallTimes[i]))
        print("Median:", np.median(overallTimes[i]))
        print(len(overallTimes[i]))
print()
print("Optimal Time")
print("Min:", np.min(optimalTimes))
print("Max:", np.max(optimalTimes))
print("Mean:", np.mean(optimalTimes))
print("Median:", np.median(optimalTimes))
print(len(optimalTimes))
print()

gravesTimes = json.load(open("graph_builder_times.json"))
gravesTimes = {y.split("/")[-1]+".json|||"+str(i):gravesTimes[y] for y in gravesTimes for i in range(4)}

gravesScores = json.load(open("graves/aggregators=[max_mean_std]_edge_sets=[ICFG_Data]_epochs=25_mode=cat_mp_layers=4_net=EGC_no_jump=True_pool_type=[max_mean_attention]_problem_types=[termination_overflow_reachSafety_memSafety]_task=rank_1663238934842323694.json"))

gravesOverallTimes = []
gravesTop4OverallTimes = []
overheads = []
verificationTimes = []
percent = []
TO=0
for problem in labels:
    res, time = gravesScores[problem]
    scores = [x[0] for x in labels[problem]]
    times = [(x[0]-x[1])*1000 for x in labels[problem]]
    if scores[np.argmax(res)] >0 and times[np.argmax(res)]+time+gravesTimes[problem] <900 :
        gravesOverallTimes.append(times[np.argmax(res)]+time+gravesTimes[problem])
        verificationTimes.append(times[np.argmax(res)])
        overheads.append(time+gravesTimes[problem])
        percent.append(time+gravesTimes[problem]/(times[np.argmax(res)]+time+gravesTimes[problem]))
    elif scores[np.argmax(res)] >0 and times[np.argmax(res)]<900:
        TO+=1
    
    minVal = 1000
    for i in np.argsort(times)[:4]:
        if scores[i] >0 and times[i]< minVal:
            minVal = times[i]
    gravesTop4OverallTimes.append(minVal+time+gravesTimes[problem])

print("Graves Time")
print("Min:", np.min(gravesOverallTimes))
print("Max:", np.max(gravesOverallTimes))
print("Mean:", np.mean(gravesOverallTimes))
print("Median:", np.median(gravesOverallTimes))
print(len(gravesOverallTimes))
print(TO)

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

y = [np.sum(np.array(cstOverallTimes)<j) for j in threshholds]
plt.plot(y, threshholds, label="CST", marker='d')

optimalTimes.sort()
y = [np.sum(np.array(optimalTimes)<j) for j in threshholds]
plt.plot(y, threshholds, label="VBS", marker='X')

markers = ['v','^','s','p','+','x','D','*','1',8]
colors=['lightcoral', 'peru', 'gold', 'darkseagreen', 'aquamarine','teal','lightsteelblue','darkviolet','orchid', 'crimson']
verifiers = {0:"CPA-Seq", 1:"2LS", 2:"ESBMC-incr", 3:"DepthK", 4:"UTaipan",5:"Symbiotic", 6:"ESBMC-kind", 7:'UKojak',8:'CBMC',9:'UAutomizer'}
x = list(verifiers.items())
x.sort(key=lambda x : x[1])

for i, j in x:
    y = [np.sum(np.array(overallTimes[i])<j) for j in threshholds]
    plt.plot(y, threshholds, label=j, marker=markers[i], color=colors[i])

plt.grid()
plt.xlabel("Problems Solved")
plt.ylabel("Time (s)")
plt.yscale('log')
plt.legend(loc=2, prop={'size': 5.5})
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