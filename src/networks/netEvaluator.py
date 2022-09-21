from gnn import EGC
from utils.utils import ModifiedMarginRankingLoss, getCorrectProblemTypes, evaluate, GeometricDataset
import torch, json, time, argparse
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GNNExplainer, AttentionalAggregation
import torch_geometric
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GNN Evaluator")
    parser.add_argument("--network", required=True, help="The network being evaluated")
    parser.add_argument("--test", required=True, nargs=1, help="The test set")
    parser.add_argument("--dataset", nargs=1, help="Directory housing the dataset")
    parser.add_argument("-p", "--problem-types", help="Which problem types to consider:termination, overflow, reachSafety, memSafety (Default=All)", nargs="+", default=['termination', 'overflow', 'reachSafety', 'memSafety'], choices=['termination', 'overflow', 'reachSafety', 'memSafety'])

    args = parser.parse_args()


    try:
        testFiles = json.load(open(args.test[0]))
    except:
        print("Error:", args.test[0], "does not exists. Please input a valid file")
        exit(1)
    testLabels = [(key, [item[1] for item in testFiles[key]]) for key in testFiles]
    testLabels = getCorrectProblemTypes(testLabels, args.problem_types)

    netName = os.path.basename(args.network)
    mp_layers = int(netName.split("mp_layers=")[1][0])
    aggregators = netName.split("aggregators=[")[1].split("]")[0].split("_")
    pools = netName.split("pool_type=[")[1].split("]")[0].split("_")
    edge_sets = netName.split("edge_sets=[")[1].split("]")[0].split("_")

    model = EGC(passes=mp_layers, inputLayerSize=155, outputLayerSize=10, pool=pools, aggregators=aggregators).cuda()

    model.load_state_dict(torch.load(args.network))

    test_set = GeometricDataset(testLabels, args.dataset[0], edge_sets, should_cache=False)


    (overallRes, overflowRes, reachSafetyRes, terminationRes, memSafetyRes), (overallChoices, overflowChoices, reachSafetyChoices, terminationChoices, memSafetyChoices) = evaluate(model, test_set, files=[x for x in testLabels])