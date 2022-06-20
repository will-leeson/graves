from operator import pos
from numpy.core.fromnumeric import sort
import torch_geometric
from torch_geometric.data import Dataset as GDataset
from torch_geometric.data import Data
from torch.nn import MarginRankingLoss
import torch
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import os, itertools, tqdm, json, time
import numpy as np
from scipy.stats import spearmanr
from torch.distributed import ReduceOp
from torch.cuda.amp import autocast

'''
File - utils.py
This file defines some utility functions
'''

class GeometricDataset(GDataset):
    def __init__(self, labels, data_dir, edge_sets, should_cache=False):
        self.labels = labels
        self.data_dir = data_dir
        self.edge_sets = edge_sets
        if should_cache:
            self.cache = dict()
        else:
            self.cache = None 
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        res = None
        if (not self.cache) or (idx not in self.cache):
            path = os.path.join(self.data_dir, self.labels[idx][0].split("|||")[0]+".npz")
            edges = np.load(os.path.join(self.data_dir, self.labels[idx][0].split("|||")[0]+"Edges.npz"))

            edges_tensor = [torch.from_numpy(edges[edgeSet]) for edgeSet in self.edge_sets]

            edge_labels = torch.cat([torch.full((len(edges_tensor[i]),1),i) for i in range(len(edges_tensor))], dim=0).float()        
            edges_tensor = torch.cat(edges_tensor).transpose(0,1).long()

            data = np.load(path)
            tokens = torch.from_numpy(data['node_rep'])

            label = torch.tensor(self.labels[idx][1])
            problemType = torch.tensor([float(self.labels[idx][0].split("|||")[1])])

            res = Data(x=tokens.float(), edge_index=edges_tensor, edge_attr=edge_labels, problemType=problemType),label

        if self.cache and idx not in self.cache:
            self.cache[idx] = res
        elif self.cache:
            res = self.cache[idx]

        return res

class ModifiedMarginRankingLoss(nn.Module):
    def __init__(self, margin=0,gpu=0):
        super(ModifiedMarginRankingLoss, self).__init__()
        self.margin=margin
        self.gpu = gpu
    
    def forward(self, scores, labels):
        loss = torch.zeros(1).to(device=self.gpu)
        for i, j in itertools.combinations(list(range(len(labels[0]))),2):
            loss_fn = MarginRankingLoss(margin=self.margin*abs(i-j))
            indx = labels.argsort()
            loss += loss_fn(scores.gather(1, indx[:,i].unsqueeze(1)), scores.gather(1, indx[:,j].unsqueeze(1)), torch.tensor([[1 if i > j else -1]*scores.size(0)]).to(device=self.gpu))
        return loss

class topKLoss(nn.Module):
    def __init__(self, k=1):
        super(topKLoss, self).__init__()
        self.lossFn = nn.MSELoss()
        self.k=k

    def forward(self, scores, labels):
        topKVals, topKIdxs = labels.topk(self.k)

        return self.lossFn(scores[:,topKIdxs.squeeze(0)], topKVals)



def train_model(model, loss_fn, batchSize, trainset, valset, optimizer, scheduler, num_epochs, gpu, task, k=1, trainWeights=None, valWeights=None):
    '''
    Function used to train networks
    '''
    if trainWeights is None:
        train_loader = torch_geometric.loader.DataLoader(dataset=trainset, batch_size=batchSize, shuffle=True)
        val_loader = torch_geometric.loader.DataLoader(dataset=valset, batch_size=batchSize, shuffle=True)
    else:
        trainSampler = WeightedRandomSampler(weights=trainWeights, num_samples=len(trainWeights))
        valSampler = WeightedRandomSampler(weights=valWeights, num_samples=len(valWeights))

        train_loader = torch_geometric.loader.DataLoader(dataset=trainset, batch_size=batchSize, sampler=trainSampler)
        val_loader = torch_geometric.loader.DataLoader(dataset=valset, batch_size=batchSize, sampler=valSampler)

    train_accuracies = []; val_accuracies = []
    train_losses = []; val_losses = []

    for epoch in range(0, num_epochs):
        corr_sum = 0.0
        cum_loss = 0.0

        topk_acc = 0.0
        success_acc = 0.0
        model.train()
        torch.enable_grad()
        success_counter = 0
        allocated = []
        for (i, (graphs, labels)) in enumerate(tqdm.tqdm(train_loader)):
            allocated.append(torch.cuda.memory_allocated())
            graphs = graphs.to(device=gpu)
            labels = labels.to(device=gpu)
            if task == "success" and labels.max()<0:
                pass
            success_counter+=1

            with autocast():
                scores = model(graphs.x, graphs.edge_index, graphs.edge_attr, graphs.problemType, graphs.batch)
                if task == "rank":
                    loss = loss_fn(scores, labels)
                    cum_loss+=loss.cpu().detach().item()
                elif task == "topk" or task == "success":
                    loss = loss_fn(nn.functional.log_softmax(scores, dim=1), labels.argmax(dim=1))    
                    cum_loss+=loss.cpu().detach().item()


            for j in range(len(labels)):
                corr, _ = spearmanr(labels[j].cpu().detach(), scores[j].cpu().detach().tolist())
                corr_sum+=corr
                assert corr <=1, str(scores) + " " + str(labels)
                _, scoreTopk = scores.topk(k)
                labelTopk = labels.argmax()
                topk_acc += labelTopk in scoreTopk
                
                success_acc += 1 if labels[j][scores.argmax()]>0 else 0


            optimizer.zero_grad()
            loss.backward()
            model.float()
            optimizer.step()

            if (((i+1)/round(len(trainset), -2))*100)%10==0 or (i+1)==len(train_loader):
                mystr = "Train-epoch "+ str(epoch) + ", Avg-Loss: "+ str(round(cum_loss/(i*batchSize), 4)) + ", Avg-Corr:" +  str(round(corr_sum/(i*batchSize), 4)) + ", TopK-Acc:"+str(round(topk_acc/(i*batchSize), 4)) + ", Success-Acc:"+str(round(success_acc/success_counter,4))
                print(mystr)
                train_accuracies.append(round(corr_sum/i, 4))
                train_losses.append(round(cum_loss/i, 4))

        corr_sum = 0.0
        cum_loss = 0.0
        model.eval()

        topk_acc = 0.0
        topk_loss = 0.0
        success_acc = 0.0
        success_counter = 0

        for (i, (graphs,labels)) in enumerate((val_loader)):
            graphs = graphs.to(device=gpu)
            labels = labels.to(device=gpu)
            if task == "success" and labels.max()<0:
                pass
            success_counter+=1
            with autocast():
                with torch.no_grad():
                    scores = model(graphs.x, graphs.edge_index, graphs.edge_attr, graphs.problemType, graphs.batch)
                    if task == "rank":
                        loss = loss_fn(scores, labels)
                    elif task == "topk" or task == "success":
                        loss = loss_fn(nn.functional.log_softmax(scores, dim=1), labels.argmax(dim=1))    
                    cum_loss+=loss.cpu().detach().item()

            for j in range(len(labels)):
                corr, _ = spearmanr(labels[j].cpu().detach(), scores[j].cpu().detach().tolist())
                corr_sum += corr
                _, scoreTopk = scores.topk(k)
                labelTopk = labels.argmax()
                topk_acc += labelTopk in scoreTopk

                success_acc += 1 if labels[j][scores.argmax()]>0 else 0

        scheduler.step(cum_loss/(i+1))


        val_accuracies.append(round(corr_sum/i, 4))
        val_losses.append(round(cum_loss/i, 4))

        mystr = "Valid-epoch "+ str(epoch) + ", Avg-Loss: "+ str(round(cum_loss/(i*batchSize), 4)) + ", Avg-Corr:" +  str(round(corr_sum/(i*batchSize), 4)) + ", TopK-Acc:"+str(round(topk_acc/(i*batchSize), 4)) + ", Success-Acc:"+str(round(success_acc/success_counter,4))
        print(mystr)
        if optimizer.param_groups[0]['lr']<1e-7:
            break
    
    return train_accuracies, train_losses, val_accuracies, val_losses

def evaluate(model, test_set, gpu=0, k=3):
    '''
    Function used to evaluate model on test set
    '''
    corr_sum = np.array([0.0]*4)
    topKAcc = np.array([0.0]*4)
    bestPredicts = np.array([0]*4)
    correctPredicts = np.array([0]*4)
    possibleCorrect = np.array([0]*4)
    predSpot = np.array([[0]*test_set[0][1].size(0)]*4)
    probCounter = np.array([0]*4)

    predicted = np.array([[0]*test_set[0][1].size(0)]*5)

    model.eval()

    test_loader = torch_geometric.loader.DataLoader(dataset=test_set, batch_size=1)

    for (i, (graphs,labels)) in enumerate(tqdm.tqdm(test_loader, leave=False)):
        graphs = graphs.to(device=gpu)
        labels = labels.to(device=gpu)
        problemTypes = graphs.problemType
        try:
            with autocast():
                with torch.no_grad():
                    scores = model(graphs.x, graphs.edge_index, graphs.edge_attr, graphs.problemType, graphs.batch)
        except RuntimeError:
            time.sleep(2)
            continue

        for j in range(len(labels)):
            corr, _ = spearmanr(labels[j].cpu().detach(), scores[j].cpu().detach().tolist())
            corr_sum[int(problemTypes.item())] += corr
            _, scoreTopk = scores.topk(k)
            labelTopk = labels.argmax()
            topKAcc[int(problemTypes.item())] += labelTopk in scoreTopk

        bestPredicts[int(problemTypes.item())] += (scores.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

        for idx in scores.argmax(dim=1):
            predicted[0][idx.item()]+=1
            predicted[int(problemTypes.item())+1][idx.item()]+=1

        maxScoresIdx = scores.argmax(dim=1).reshape(len(scores),1)
        gather = labels.gather(1, maxScoresIdx)
        if labels.min()<=0:
            correctPredicts[int(problemTypes.item())]+=(gather>0).sum().item()

        predSpot[int(problemTypes.item())][np.where((-labels).argsort().cpu().numpy()==scores.argmax().item())[1]] +=1
        
        
        if labels.max() > 0 and labels.min()<=0:
            possibleCorrect[int(problemTypes.item())]+=1
        probCounter[int(problemTypes.item())]+=1
    
    res = [[],[],[],[],[]]
    
    res[0] = np.array([corr_sum.sum()/probCounter.sum(), topKAcc.sum()/probCounter.sum(), bestPredicts.sum()/probCounter.sum(), correctPredicts.sum()/possibleCorrect.sum(), predSpot.sum(axis=0)], dtype=object)
    for i in range(0,len(res)-1):
        res[i+1] = np.array([corr_sum[i]/probCounter[i], topKAcc[i]/probCounter[i], bestPredicts[i]/probCounter[i], correctPredicts[i]/possibleCorrect[i], predSpot[i]], dtype=object)


    return res, predicted
def getCorrectProblemTypes(labels, problemTypes):
    '''
    Function used to make sure we are only looking at problem types that we want
    '''
    if "overflow" not in problemTypes:
	    labels = [item for item in labels if item[0].split("|||")[1]!="0"]
    if "reachSafety" not in problemTypes:
    	labels = [item for item in labels if item[0].split("|||")[1]!="1"]
    if "termination" not in problemTypes:
        labels = [item for item in labels if item[0].split("|||")[1]!="2"]
    if "memSafety" not in problemTypes:
        labels = [item for item in labels if item[0].split("|||")[1]!="3"]

    return labels

def groupLabels(labels, mapping="../../data/toolMapping.json"):
    toolToAlg, algToTool = json.load(open(mapping))

    newLabels = []
    for label in labels:
        vals = []
        for alg in algToTool:
            val = []
            for tool in algToTool[alg]:
                val.append(label[1][toolToAlg[tool][0]]/len(toolToAlg[tool][1]))
            vals.append(sum(val)/len(algToTool[alg]))
        newLabels.append((label[0],vals))

    assert(len(newLabels) == len(labels))
    return newLabels

def getWeights(train_set):
    labels = np.array([x[1] for x in train_set])
    weightDict = {tuple(t):1./sum(np.all(labels.argsort() == t, axis=1)) for t in np.unique(labels.argsort(),axis=0)}

    return [weightDict[tuple(t)] for t in labels.argsort()]
