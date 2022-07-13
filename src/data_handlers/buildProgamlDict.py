import programl as pg
import json, tqdm
import multiprocessing as mp
import re

train = json.load(open("../../data/subsetTrainFiles.json"))
val = json.load(open("../../data/subsetValFiles.json"))
files = open("../../data/programs.txt").readlines()

allowed = list(train.keys())+list(val.keys())
allowed = [x[:-9] for x in allowed]
files = ["../"+file.strip()[:-2] +".ll" for file in files if file.strip().split("/")[-1] in allowed]

def handler(file):
    nodes = set()
    progFile = open(file)

    prog = "".join(progFile.readlines())

    try:
        G = pg.from_llvm_ir(prog)
    except Exception as e:
        print(e)
        print(file)
        return set()

    for node in G.node:
        if "[" in node.text and " x " in node.text:
            x = re.findall(r'[0-9]+\ x',node.text)
            x.sort(key=lambda x: len(x), reverse=True)
            for item in x:
                node.text = node.text.replace(item, "array of")
        if "%struct" in node.text  or "%\"struct" in node.text:
            x = re.findall(r'%["]?struct\.[a-zA-Z0-9_$]+["]?[\*]*',node.text)
            for item in x:
                node.text = node.text.replace(item, "%struct")
        if "%union" in node.text:
            x = re.findall(r'%union\.[a-zA-Z0-9_$]+[\*]*', node.text)

            for item in x:
                node.text = node.text.replace(item, "%union")
        
        nodes.add(node.text)
    
    return nodes

pool = mp.Pool(processes=mp.cpu_count()-2)
jobs = []

for file in files:
    jobs.append(pool.apply_async(handler, args=([file])))

finalNodes = set()
for job in tqdm.tqdm(jobs):
    finalNodes = finalNodes.union(job.get())

nodeDict = {node:i for i,node in enumerate(finalNodes)}
nodeDict["unk"] = {len(nodeDict)}
json.dump(nodeDict, open("nodeDict.json",'w'))