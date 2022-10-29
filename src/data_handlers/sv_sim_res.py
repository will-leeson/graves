import glob, tqdm, os
import json

specs = ["Reach", "Overflow", "Termination", "MemSafety", "Concurrency"]
propertyFiles = {"Reach":"properties/unreach-call.prp", "Overflow":"properties/no-overflow.prp", "Termination":"properties/termination.prp", "MemSafety":"properties/valid", "Concurrency":"properties/no-data-race.prp"}

falsificationScore = {"Graves":0, "Graves-GAT":0, "PeSCo":0}
overallScore = {"Graves":0, "Graves-GAT":0, "PeSCo":0}
softwareSystemsScore = {"Graves":0, "Graves-GAT":0, "PeSCo":0}

falsificationProblems = 0
overallProblems = 0
softwareSystemsProblems = 0

overallCategories = 0
softwareSystemsCategories = 0
falsificationCategories = 0

for spec in specs:
    print(spec)
    reachFiles = {x : dict() for x in glob.glob("/scratch/wel2vw/sv-benchmarks/c/*"+spec+"*-*.set")}
    softwareReachFiles = {x : dict() for x in glob.glob("/scratch/wel2vw/sv-benchmarks/c/SoftwareSystems-*-"+spec+"*.set")}

    foundProp = False
    for someFiles in [reachFiles, softwareReachFiles]:
        for reachSet in tqdm.tqdm(someFiles):
            for line in open(reachSet):
                files = glob.glob("/scratch/wel2vw/sv-benchmarks/c/"+line.strip())
                for file in files:
                    if os.path.isdir(file):
                        continue
                    for line2 in open(file):
                        if foundProp:
                            fileName = "/".join(file.split("/")[-2:])
                            if "expected_verdict: true" in line2.lower():
                                someFiles[reachSet][fileName] = True
                            elif "expected_verdict: false" in line2.lower():
                                someFiles[reachSet][fileName] = False
                            else:
                                print(line2)
                                assert False, "Couldn't find expected verdict"
                            foundProp = False
                            break
                                
                        foundProp =  propertyFiles[spec] in line2
                        
    rawRes = json.load(open("/scratch/wel2vw/sv_sim.json"))

    reachRes = dict()

    wrong = {"Graves":0, "Graves-GAT":0, "PeSCo":0}
    correct = {"Graves":0, "Graves-GAT":0, "PeSCo":0}
    unknown = {"Graves":0, "Graves-GAT":0, "PeSCo":0}

    scores1 = {key : {x : [] for x in glob.glob("/scratch/wel2vw/sv-benchmarks/c/*"+spec+"*-*.set")} for key in wrong}

    for reachSet in reachFiles:
        for problem in reachFiles[reachSet]:
            if problem in rawRes[spec]:
                if reachFiles[reachSet][problem]:
                    for key in wrong:
                        score = 0
                        if key not in rawRes[spec][problem] or rawRes[spec][problem][key][0] == "Unknown":
                            unknown[key]+=1
                        elif rawRes[spec][problem][key][0]:
                            score=2
                            correct[key]+=1
                        else:
                            score=-16
                            wrong[key]+=1
                        scores1[key][reachSet].append(score)
                else:
                    for key in wrong:
                        score = 0
                        if key not in rawRes[spec][problem] or rawRes[spec][problem][key][0] == "Unknown":
                            unknown[key]+=1
                        elif rawRes[spec][problem][key][0]:
                            score=-32
                            wrong[key]+=1
                        else:
                            score=1
                            correct[key]+=1
                        scores1[key][reachSet].append(score)

    for key in wrong:
        print(key," correct:", correct[key])
        print(key," wrong:", wrong[key])
        print(key," unknown:", unknown[key])


    catgeoryScoring = {"Graves":dict(), "PeSCo":dict(), "Graves-GAT":dict()}
    scores = {"Graves":0, "PeSCo":0, "Graves-GAT":0}

    totalProblems = 0
    for category in reachFiles:
        pescoScore = 0
        gravesScore = 0
        numProblems = len(reachFiles[category])
        totalProblems += numProblems
        
        if spec == "Reach":
            for key in catgeoryScoring:
                catgeoryScoring[key][category]=sum(scores1[key][category])
                scores[key] += sum(scores1[key][category])/numProblems
                overallScore[key]+= sum(scores1[key][category])/numProblems
                falsificationScore[key] +=sum(scores1[key][category])/numProblems
                falsificationProblems+=numProblems
                falsificationCategories +=1
        else:
            for key in catgeoryScoring:
                catgeoryScoring[key][category]=sum(scores1['Graves-GAT'][category])
                scores[key] += sum(scores1['Graves-GAT'][category])/numProblems
                overallScore[key]+= sum(scores1['Graves-GAT'][category])/numProblems
                if spec != "Termination":
                    falsificationScore[key] +=sum(scores1['Graves-GAT'][category])/numProblems
                    falsificationProblems+=numProblems
                    falsificationCategories += 1


    overallProblems += totalProblems
    overallCategories += len(reachFiles)
    print(totalProblems)

    for key in scores:
        print(key,spec, "Score:",scores[key]*totalProblems/len(reachFiles))
    
    wrong = {"Graves":0, "Graves-GAT":0, "PeSCo":0}
    correct = {"Graves":0, "Graves-GAT":0, "PeSCo":0}
    unknown = {"Graves":0, "Graves-GAT":0, "PeSCo":0}

    scores1 = {key : {x : [] for x in glob.glob("/scratch/wel2vw/sv-benchmarks/c/SoftwareSystems-*-"+spec+"*.set")} for key in wrong}

    for reachSet in softwareReachFiles:
        for problem in softwareReachFiles[reachSet]:
            if problem in rawRes[spec]:
                if softwareReachFiles[reachSet][problem]:
                    for key in wrong:
                        score = 0
                        if key not in rawRes[spec][problem] or rawRes[spec][problem][key][0] == "Unknown":
                            unknown[key]+=1
                        elif rawRes[spec][problem][key][0]:
                            score=2
                            correct[key]+=1
                        else:
                            score=-16
                            wrong[key]+=1
                        scores1[key][reachSet].append(score)
                else:
                    for key in wrong:
                        score = 0
                        if key not in rawRes[spec][problem] or rawRes[spec][problem][key][0] == "Unknown":
                            unknown[key]+=1
                        elif rawRes[spec][problem][key][0]:
                            score=-32
                            wrong[key]+=1
                        else:
                            score=1
                            correct[key]+=1
                        scores1[key][reachSet].append(score)

    for key in wrong:
        print("Software Systems")
        print(key," correct:", correct[key])
        print(key," wrong:", wrong[key])
        print(key," unknown:", unknown[key])

    catgeoryScoring = {"Graves":dict(), "PeSCo":dict(), "Graves-GAT":dict()}
    scores = {"Graves":0, "PeSCo":0, "Graves-GAT":0}

    totalProblems = 0
    for category in softwareReachFiles:
        pescoScore = 0
        gravesScore = 0
        numProblems = len(softwareReachFiles[category])
        totalProblems += numProblems
        if numProblems <= 0:
            continue       
        if spec == "Reach":
            for key in catgeoryScoring:
                catgeoryScoring[key][category]=sum(scores1[key][category])
                scores[key] += sum(scores1[key][category])/numProblems
                overallScore[key]+= sum(scores1[key][category])/numProblems
                softwareSystemsScore[key]+=sum(scores1[key][category])/numProblems
                falsificationScore[key] +=sum(scores1[key][category])/numProblems
        else:
            for key in catgeoryScoring:
                catgeoryScoring[key][category]=sum(scores1['Graves-GAT'][category])
                scores[key] += sum(scores1['Graves-GAT'][category])/numProblems
                overallScore[key]+= sum(scores1['Graves-GAT'][category])/numProblems
                softwareSystemsScore[key]+=sum(scores1['Graves-GAT'][category])/numProblems
                falsificationScore[key] +=sum(scores1['Graves-GAT'][category])/numProblems
        
        falsificationProblems+=totalProblems
        overallProblems += totalProblems
        softwareSystemsProblems += totalProblems
        overallCategories += len(softwareReachFiles)
        softwareSystemsCategories += len(softwareReachFiles)
        falsificationCategories += len(softwareReachFiles)
    
    for key in softwareSystemsScore:
        print(key,spec, "Score:",softwareSystemsScore[key]*totalProblems/len(reachFiles))

print()

for key in softwareSystemsScore:
    print(key,"Software Systems Score:",softwareSystemsScore[key]*softwareSystemsProblems/softwareSystemsCategories)
    
print()

for key in falsificationScore:
    print(key,"Falsification Score:",falsificationScore[key]*falsificationProblems/falsificationCategories)

print()

for key in overallScore:
    print(key,"Overall Score:",overallScore[key]*overallProblems/overallCategories)