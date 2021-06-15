import glob, csv, os, json

'''
File - resultCompiler.py

This file will look through the sv-comp results to
produce the labels for files
'''

solFiles = glob.glob("../../data/benchexecOutput/*/*.csv")
results = dict()

key = []
for tool in solFiles:
    key.append(tool)
    reader = csv.reader(open(tool), delimiter="\t")
    #skip headers
    counter = 0
    for i in range(3):
        next(reader)
    toolName = os.path.basename(tool).strip(".csv")
    toolResults = dict()
    
    for row in reader:
        result = list(filter(lambda a: a!="", row))
        name = os.path.basename(result[0])+".json"
        score = None
        category = -1
        if "unreach-call" in result[0]:
            category = 1
            if "true" not in result[1] and "false" not in result[1]:
                score = 0
            elif "true" in result[1]:
                if "true-unreach-call" in name:
                    score = 2
                else:
                    score = -1
            elif "false" in result[1]:
                if "false-unreach-call" in name:
                    score = 1
                else:
                    score = -2
            else:
                assert()

        assert(category!=-1)
        assert(score is not None)

        if name+"|||"+str(category) in results:
            results[name+"|||"+str(category)].append([score, float(score)-(float(result[2])/1000)])
        else:
            results[name+"|||"+str(category)] = [[score, float(score)-(float(result[3])/1000)]]

        # if name in toolResults:
        #     toolResults[name].append([category, score, float(score)-(float(result[3])/1000), float(score)-(float(result[5])/15000), float(score)-(float(result[6])/15000)])
        # else:
        #     toolResults[name] = [[category, score, float(score)-(float(result[3])/1000), float(score)-(float(result[5])/15000),float(score)-(float(result[6])/15000)]]
    # results[toolName] = toolResults

for item in results:
    results[item].append([0,0])

reader = csv.reader(open(solFiles[0]), delimiter="\t")
programs = open("../../data/benchExecPrograms.txt", 'w')
programsList = []
#skip headers
for i in range(3):
    next(reader)
for row in reader:
    if row[0] in programsList:
        continue
    else:
        programs.write("../data/sv-benchmarks/c/"+row[0]+"\n")
        programsList.append(row[0])

json.dump(results, open("../../data/BenchExecResults.json",'w'))
keyFile = open("../../data/benchExecKey.txt", 'w')
for item in key:
    keyFile.write(os.path.basename(item).strip(".csv")+"\n")
keyFile.write("Unknown")