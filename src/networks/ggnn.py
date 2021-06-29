import torch
import torch.nn as nn
import torch.nn.functional as f

'''
File - ggnn.py

This file includes three architectures for GGNNs, two of which do not
actually use gated recurrent units.
'''

class GGNN(nn.Module):
    '''
    Base GGNN class
    This Graph Neural Network includes a GRU to produce new node representations
    '''
    def __init__(self, passes, numEdgeSets, inputLayerSize, outputLayerSize):
        super(GGNN, self).__init__()
        self.passes = passes
        self.gru = nn.GRUCell(150, 150)
        self.edgeNets = []
        for _ in range(numEdgeSets):
           self.edgeNets.append([nn.Linear(inputLayerSize,inputLayerSize).cuda(),nn.Linear(inputLayerSize,inputLayerSize).cuda()])
        self.fc1 = nn.Linear(inputLayerSize+1, 80)
        self.fc2 = nn.Linear(80,80)
        self.fcLast = nn.Linear(80, outputLayerSize)

    def forward(self, nodesBatch, backwards_edgeBatch, problemTypeBatch):
        for j in range(self.passes):
            for i in range(len(nodesBatch)):
                incoming = torch.zeros_like(nodesBatch[i])
                counter = 0
                for edgeSet in backwards_edgeBatch[i]:
                    try:
                        y = self.edgeNets[counter][0](nodesBatch[i][edgeSet[:,1]])
                        y = f.leaky_relu(y)
                        y = self.edgeNets[counter][1](y)
                        incoming = incoming.index_add(0, edgeSet[:,0].cuda(), y)
                    except IndexError:
                        continue #Empty Edge Set
                    counter+=1
                nodesBatch[i] = self.gru(incoming, nodesBatch[i])
                if j < self.passes - 2:
                    nodesBatch[i] = nodesBatch[i].detach()
        
        for i in range(len(nodesBatch)):
                nodesBatch[i] = nodesBatch[i].sum(dim=0)
                nodesBatch[i] = torch.log(nodesBatch[i])
                nodesBatch[i][torch.isnan(nodesBatch[i])] = 0
                nodesBatch[i] = f.relu(nodesBatch[i])
                nodesBatch[i][nodesBatch[i]==float("inf")] = nodesBatch[i][nodesBatch[i]!=float("inf")].max()
        
        x = torch.stack(nodesBatch)
        x = torch.cat((x, problemTypeBatch), dim=1)
        x = self.fc1(x)
        x = f.leaky_relu(x)
        x = self.fc2(x)
        x = f.leaky_relu(x)
        x = self.fcLast(x)
        return x

class GGNN_NoGRU(nn.Module):
    '''
    GGNN without GRU, so more aptly a GNN
    '''
    def __init__(self, passes, numEdgeSets, inputLayerSize, outputLayerSize):
        super(GGNN_NoGRU, self).__init__()
        self.passes = passes
        self.edgeNets = []
        for i in range(numEdgeSets):
           self.edgeNets.append([nn.Linear(inputLayerSize,inputLayerSize).cuda(),nn.Linear(inputLayerSize,inputLayerSize).cuda()])
        self.fc1 = nn.Linear(inputLayerSize+1, 80)
        self.fc2 = nn.Linear(80,80)
        self.fcLast = nn.Linear(80, outputLayerSize)

    def forward(self, nodesBatch, backwards_edgeBatch, problemTypeBatch):
        for j in range(self.passes):
            for i in range(len(nodesBatch)):
                incoming = torch.zeros_like(nodesBatch[i]).cuda()
                counter = 0
                for edgeSet in backwards_edgeBatch[i]:
                    try:
                        y = self.edgeNets[counter][0](nodesBatch[i][edgeSet[:,1]])
                        y = f.leaky_relu(y)
                        y = self.edgeNets[counter][1](y)
                        incoming = incoming.index_add(0, edgeSet[:,0].cuda(), y)
                    except IndexError:
                        pass #Empty Edge Set
                    counter+=1
                nodesBatch[i] = nodesBatch[i] + incoming*(self.passes-j)/self.passes
        for i in range(len(nodesBatch)):
                nodesBatch[i] = nodesBatch[i].sum(dim=0)
                nodesBatch[i] = torch.log(nodesBatch[i])
                nodesBatch[i][torch.isnan(nodesBatch[i])] = 0
                nodesBatch[i] = f.relu(nodesBatch[i])
                nodesBatch[i][nodesBatch[i]==float("inf")] = nodesBatch[i][nodesBatch[i]!=float("inf")].max()
        
        x = torch.stack(nodesBatch)
        x = torch.cat((x, problemTypeBatch), dim=1)
        x = self.fc1(x)
        x = f.leaky_relu(x)
        x = self.fc2(x)
        x = f.leaky_relu(x)
        x = self.fcLast(x)
        return x

class GGNN_NoGRU_NoEdgeNets(nn.Module):
    '''
    Base GGNN with GRU or edgenets
    '''
    def __init__(self, passes, numEdgeSets, inputLayerSize, outputLayerSize):
        super(GGNN_NoGRU_NoEdgeNets, self).__init__()
        self.passes = passes
        self.fc1 = nn.Linear(inputLayerSize+1, 80)
        self.fc2 = nn.Linear(80,80)
        self.fcLast = nn.Linear(80, outputLayerSize)

    def forward(self, nodesBatch, backwards_edgeBatch, problemTypeBatch):  
        for j in range(self.passes):
            for i in range(len(nodesBatch)):
                incoming = torch.zeros_like(nodesBatch[i])
                counter = 0
                for edgeSet in backwards_edgeBatch[i]:
                    try:
                        incoming = incoming.index_add(0, edgeSet[:,0].cuda(), nodesBatch[i][edgeSet][:,1])
                    except IndexError:
                        continue #Empty Edge Set
                    counter+=1
                nodesBatch[i] = nodesBatch[i] + incoming
        
        for i in range(len(nodesBatch)):
                nodesBatch[i] = nodesBatch[i].sum(dim=0)
                nodesBatch[i] = torch.log(nodesBatch[i])
                nodesBatch[i][torch.isnan(nodesBatch[i])] = 0
                nodesBatch[i] = f.relu(nodesBatch[i])
                nodesBatch[i][nodesBatch[i]==float("inf")] = nodesBatch[i][nodesBatch[i]!=float("inf")].max()
        
        x = torch.stack(nodesBatch)
        x = torch.cat((x, problemTypeBatch), dim=1)
        x = self.fc1(x)
        x = f.leaky_relu(x)
        x = self.fc2(x)
        x = f.leaky_relu(x)
        x = self.fcLast(x)
        return x
