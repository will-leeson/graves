import glob, csv, os, re
import numpy as np

resultsFiles = glob.glob("../../data/trainedNetsResults/*.npz")

rawDataFile = csv.writer(open("rawData.csv", 'w'), delimiter=",")
collatedDataFile = csv.writer(open("collatedData.csv", 'w'), delimiter=",")

collatedData = dict()
rawDataFile.writerow(["numPasses", "hasGru", "hasEdgeNets", "trainginData", "hasAST", "hasCFG", "hasDFG", "corr", "bestPredicts", "correctPredicts"])
for aFile in resultsFiles:
    numPasses = re.findall(r'[0-9]+', os.path.basename(aFile))[0]
    assert(int(numPasses)<=5)
    hasGru = "GRU" not in aFile
    hasEdgeNets = "EdgeNets" not in aFile
    hasAST = "NoAST" not in aFile
    hasCFG = "NoCFG" not in aFile
    hasDFG = "NoDFG" not in aFile
    trainingData = "all"
    if "overflow" in aFile:
        trainingData = "overflow"
    elif "memSafety" in aFile:
        trainingData = "memSafety"
    elif "reachSafety" in aFile:
        trainingData = "reachSafety"
    elif "termination" in aFile:
        trainingData = "termination"
    
    data = np.load(aFile, allow_pickle=True)['arr_4']
    if len(data)==3:
        corr, bestPredicts, correctPredicts = data
    else:
        corr, bestPredicts, correctPredicts, _, _ = data
    rawDataFile.writerow([numPasses, hasGru, hasEdgeNets, hasAST, hasCFG, hasDFG, corr, bestPredicts, correctPredicts])
    if (numPasses, hasGru, hasEdgeNets, trainingData, hasAST, hasCFG, hasDFG) in collatedData:
        collatedData[(numPasses, hasGru, hasEdgeNets, trainingData, hasAST, hasCFG, hasDFG)].append([corr, bestPredicts, correctPredicts]) 
    else:
        collatedData[(numPasses, hasGru, hasEdgeNets, trainingData, hasAST, hasCFG, hasDFG)] = [[corr, bestPredicts, correctPredicts]]

collatedDataFile.writerow(["numPasses", "hasGru", "hasEdgeNets", "trainginData", "hasAST", "hasCFG", "hasDFG", "corrMean", "corrSTD", "bestPredictsMean", "bestPredictsStd", "correctPredictsMean", "correctPredictsSTD", "num"])
for key in collatedData:
    corrsMean = np.mean([x[0] for x in collatedData[key]])
    corrsStd =  np.std([x[0] for x in collatedData[key]])
    bestPredictsMean = np.mean([x[1] for x in collatedData[key]])
    bestPredictsStd =  np.std([x[1] for x in collatedData[key]])
    correctPredictsMean = np.mean([x[2] for x in collatedData[key]])
    correctPredictsStd =  np.std([x[2] for x in collatedData[key]])

    collatedData[key] = [corrsMean, corrsStd, bestPredictsMean, bestPredictsStd, correctPredictsMean, correctPredictsStd, len(collatedData[key])]

for key in collatedData:
    collatedDataFile.writerow(list(key)+collatedData[key])