import os, json, torch

'''
File - dataSplitter.py

This file will split the entire dataset into train, validation, and test sets.
It will be a 80/10/10 split, respectively
'''

resultFile = json.load(open("../../data/BenchExecResults.json"))

files = [[],[],[],[]]
for key in resultFile:
    if os.path.isfile("../../data/final_graphs/" + key.split("|||")[0]+".npz"):
        files[int(key.split("|||")[1])].append((key, resultFile[key]))

trainFiles, valFiles, testFiles = [], [], []
for i in files:
    x, y= torch.utils.data.random_split(i, [len(i) - (len(i)//10),len(i)//10])
    trainFiles+=x
    valFiles+=y

trainFileDict = dict()
valFileDict = dict()

for item in trainFiles:
    trainFileDict[item[0]] = item[1]

json.dump(trainFileDict, open("../../data/benchTrainFiles.json",'w'))

for item in valFiles:
    valFileDict[item[0]] = item[1]
json.dump(valFileDict, open("../../data/benchValFiles.json",'w'))