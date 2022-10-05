import glob, csv

analysis_files = glob.glob("*/*.txt")

codes = dict()

codeCount = {"BMC":0, "BMC+K":0, "SymEx":0, "CEGAR":0}

for code_file in analysis_files:
    alg = code_file.split("/")[0]
    code_file = open(code_file)
    for code in code_file:
        codeCount[alg]+=1
        code = code.strip()
        code = code.lower()
        if code in codes:
            if alg in codes[code]:
                codes[code][alg]+=1
            else:
                codes[code][alg]=1
        else:
            codes[code] = {alg:1}
    
csv_file = csv.writer(open("open_coding.csv",'w'), delimiter=",")
csv_file.writerow(["Code","BMC","BMC+K","SymEx","CEGAR"])

for code in codes:
    toWrite = [code]
    for alg in ["BMC", "BMC+K", "SymEx", "CEGAR"]:
        if alg in codes[code]:
            toWrite.append(codes[code][alg]/codeCount[alg])
        else:
            toWrite.append(0)
    csv_file.writerow(toWrite)

print(codeCount)