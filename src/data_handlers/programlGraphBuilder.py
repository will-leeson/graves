import programl as pg
from sys import argv
import json, re
import networkx as nx
import numpy as np

progFile = open(argv[1])

prog = "".join(progFile.readlines())

nodeDict = json.load(open("../../data/nodeDict.json"))

G = pg.from_llvm_ir(prog)

nodes = []
edges = [[],[]]
edge_attr = []

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
    
    nodes.append(nodeDict[node.text])

for edge in G.edge:
    outNode, inNode, attr = int(edge.source), int(edge.target), int(edge.flow)
    assert outNode <= len(nodes) and inNode <= len(nodes)

    edges[0].append(outNode)
    edges[1].append(inNode)
    edge_attr.append(attr) 

graph = np.savez_compressed(argv[1][:-2]+"npz" ,nodes=np.array(nodes), edges=np.array(edges), edge_attr=np.array(edge_attr))