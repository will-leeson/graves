import glob, csv, os, json

'''
File - resultCompiler.py

This file will look through the sv-comp results to
produce the labels for files
'''

solFiles = glob.glob("../../data/sv-results/*.csv")
results = dict()

key = []
for tool in solFiles:
    key.append(tool)
    reader = csv.reader(open(tool), delimiter="\t")
    #skip headers
    counter = 0
    for i in range(3):
        next(reader)
    toolName = os.path.basename(tool).strip(".table.csv")
    toolResults = dict()
    
    for row in reader:
        result = list(filter(lambda a: a!="", row))
        name = os.path.basename(result[0])+".json"
        score = None
        category = -1
        if "no-overflow" in result[1]:
            category = 0
            if "true" not in result[2] and "false" not in result[2]:
                score = 0
            elif "true" in result[2]:
                if "true-no-overflow" in name:
                    score = 2
                else:
                    score = -1
            elif "false" in result[2]:
                if "false-no-overflow" in name:
                    score = 1
                else:
                    score = -2
            else:
                assert()
        elif "unreach-call" in result[1]:
            category = 1
            if "true" not in result[2] and "false" not in result[2]:
                score = 0
            elif "true" in result[2]:
                if "true-unreach-call" in name:
                    score = 2
                else:
                    score = -1
            elif "false" in result[2]:
                if "false-unreach-call" in name:
                    score = 1
                else:
                    score = -2
            else:
                assert()
        elif "termination" in result[1]:
            category = 2
            if "true" not in result[2] and "false" not in result[2]:
                score = 0
            elif "true" in result[2]:
                if "true-termination" in name:
                    score = 2
                else:
                    score = -1
            elif "false" in result[2]:
                if "false-termination" in name:
                    score = 1
                else:
                    score = -2
            else:
                assert()
        elif "valid" in result[1]:
            category = 3
            if "true" not in result[2] and "false" not in result[2]:
                score = 0
            elif "true" in result[2]:
                if "true-valid" in name:
                    score = 2
                else:
                    score = -1
            elif "false" in result[2]:
                if "false-valid" in name:
                    score = 1
                else:
                    score = -2
            else:
                assert()
        else:
            assert()

        assert(category!=-1)
        assert(score is not None)
        print(name)
        print(score)
        print(result[3])
        print(float(score)-(float(result[3])/1000))
        assert()

        if name+"|||"+str(category) in results:
            results[name+"|||"+str(category)].append([score, float(score)-(float(result[3])/1000), float(score)-(float(result[5])/15000), float(score)-(float(result[6])/15000)])
        else:
            results[name+"|||"+str(category)] = [[score, float(score)-(float(result[3])/1000), float(score)-(float(result[5])/15000), float(score)-(float(result[6])/15000)]]

        # if name in toolResults:
        #     toolResults[name].append([category, score, float(score)-(float(result[3])/1000), float(score)-(float(result[5])/15000), float(score)-(float(result[6])/15000)])
        # else:
        #     toolResults[name] = [[category, score, float(score)-(float(result[3])/1000), float(score)-(float(result[5])/15000),float(score)-(float(result[6])/15000)]]
    # results[toolName] = toolResults

reader = csv.reader(open(solFiles[0]), delimiter="\t")
programs = open("../../data/programs.txt", 'w')
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

json.dump(results, open("../../data/SV-CompResults.json",'w'))
keyFile = open("../../data/key.txt", 'w')
for item in key:
    keyFile.write(item.strip("../../data/sv-results/").strip(".table.csv")+"\n")