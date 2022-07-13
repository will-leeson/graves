import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GATv2Conv, GatedGraphConv, JumpingKnowledge, global_max_pool, global_mean_pool, global_add_pool, global_sort_pool, GlobalAttention, GraphMultisetTransformer
from torch_geometric.nn.conv.gat_conv import GATConv
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
    def __init__(self, passes, numEdgeSets, inputLayerSize, outputLayerSize, collate):
        super(GGNN, self).__init__()
        self.ggcs = nn.ModuleList([GatedGraphConv(inputLayerSize, outputLayerSize, aggr=collate) for _ in range(numEdgeSets)])
        self.passes = passes
        self.fc1 = nn.Linear(inputLayerSize+1, 80)
        self.fc2 = nn.Linear(80,80)
        self.fcLast = nn.Linear(80, outputLayerSize)
        self.collate=collate

    def forward(self, data):
        x, edge_index, problemType = data.x, data.edge_index, data.problemType

        x = f.dropout(x, p=0.6, training=self.training)

        for _ in range(self.passes):
            placeholderX = torch.zeros_like(x)
            for val, gcn in zip(torch.unique(data.edge_attr), self.ggcs):
                placeholderX += gcn(x, edge_index.transpose(0,1)[(data.edge_attr==val).squeeze()].transpose(0,1))
            x = placeholderX/len(torch.unique(data.edge_attr))

        x = f.dropout(x, p=0.6, training=self.training)
        if self.collate == "sum":
            x = global_add_pool(x, data.batch, dim=0)
        elif self.collate == "mean":
            x = global_mean_pool(x, data.batch, dim=0)
        elif self.collate == "max":
            x, _ = global_max_pool(x, data.batch, dim=0)
        else:
            raise ValueError("Not a valid collate type")

        x = torch.cat((x, problemType), dim=1)
        x = self.fc1(x)
        x = f.leaky_relu(x)
        x = self.fc2(x)
        x = f.leaky_relu(x)
        x = self.fcLast(x)
        return x

class GAT(torch.nn.Module):
    def __init__(self, passes, numEdgeSets, inputLayerSize, outputLayerSize, numAttentionLayers, mode, pool, k, shouldJump=True):
        super(GAT, self).__init__()
        self.passes = passes
        self.mode = mode
        self.k = 1
        self.shouldJump = shouldJump
            
        self.gats = nn.ModuleList([GATv2Conv(inputLayerSize,inputLayerSize, heads=numAttentionLayers, concat=False, edge_dim=1) for i in range(passes)])
        if self.passes and self.shouldJump:
           self.jump = JumpingKnowledge(self.mode, channels=inputLayerSize, num_layers=self.passes)
        if self.mode == 'cat' and self.shouldJump:
            fcInputLayerSize = ((self.passes+1)*inputLayerSize*self.k)+1
        else:
            fcInputLayerSize = (inputLayerSize*self.k)+1
        
        if pool == "add":
                self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == "sort":
            self.pool = global_sort_pool
            self.k = k
        elif pool == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(torch.nn.Linear(fcInputLayerSize-1, fcInputLayerSize//2), nn.LeakyReLU(), torch.nn.Linear(fcInputLayerSize//2, fcInputLayerSize//2), nn.LeakyReLU(), torch.nn.Linear(fcInputLayerSize//2, 1), nn.Tanh()))
        elif pool == "multiset":
            self.pool = GraphMultisetTransformer(in_channels=fcInputLayerSize-1, hidden_channels=fcInputLayerSize-1, out_channels=fcInputLayerSize-1, num_nodes=1400, num_heads=5, pool_sequences=["GMPool_I"])
        else:
            raise ValueError("Not a valid pool")

        self.fc1 = nn.Linear(fcInputLayerSize, fcInputLayerSize//2)
        self.fc2 = nn.Linear(fcInputLayerSize//2,fcInputLayerSize//2)
        self.fcLast = nn.Linear(fcInputLayerSize//2, outputLayerSize)
    
    def forward(self, x, edge_index, edge_attr, problemType, batch):
        if self.passes:
            if self.shouldJump:
                xs = [x]

            for gat in self.gats: 
                out = gat(x, edge_index, edge_attr=edge_attr)
                x = f.leaky_relu(out)
                if self.shouldJump:
                    xs += [x]

            if self.shouldJump:
                x = self.jump(xs)

        if self.pool == global_sort_pool:
            x = self.pool(x, batch, self.k)
        else:
            x = self.pool(x, batch)
        
        x = torch.cat((x.reshape(1,x.size(0)*x.size(1)), problemType.unsqueeze(1)), dim=1)
        
        x = self.fc1(x)
        x = f.leaky_relu(x)
        x = self.fc2(x)
        x = f.leaky_relu(x)
        x = self.fcLast(x)

        return x

class ProGraML_GAT(GAT):
    def __init__(self, passes, numEdgeSets, embedingDimension, outputLayerSize, numAttentionLayers, mode, pool, k, nodeDict, shouldJump=True):
        super().__init__(passes, numEdgeSets, embedingDimension, outputLayerSize, numAttentionLayers, mode, pool, k, shouldJump)
        self.nodeDict=nodeDict
        self.nodeEmbeddings = torch.randn((len(nodeDict), embedingDimension), requires_grad=True)

    def forward(self, x, edge_index, edge_attr, problemType, batch):
        x = torch.tensor([self.nodeDict[y] for y in x])
        if self.passes:
            if self.shouldJump:
                xs = [x]

            for gat in self.gats: 
                out = gat(x, edge_index, edge_attr=edge_attr)
                x = f.leaky_relu(out)
                if self.shouldJump:
                    xs += [x]

            if self.shouldJump:
                x = self.jump(xs)

        if self.pool == global_sort_pool:
            x = self.pool(x, batch, self.k)
        else:
            x = self.pool(x, batch)
        
        x = torch.cat((x.reshape(1,x.size(0)*x.size(1)), problemType.unsqueeze(1)), dim=1)
        
        x = self.fc1(x)
        x = f.leaky_relu(x)
        x = self.fc2(x)
        x = f.leaky_relu(x)
        x = self.fcLast(x)

        return x