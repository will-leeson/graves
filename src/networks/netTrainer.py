from gnn import GGNN, GAT, EGC
from utils.utils import ModifiedMarginRankingLoss, train_model, getCorrectProblemTypes, evaluate, GeometricDataset, groupLabels, getWeights
import torch, json, time, argparse
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GNNExplainer, AttentionalAggregation
import torch_geometric

'''
File - netTrainer.py
This file is a driver used to train networks
'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="GNN Trainer")
	parser.add_argument("--mp-layers", help="Number of message passing layers (Default=0)", default=0, type=int)
	parser.add_argument("-e", "--epochs", help="Number of training epochs (Default=20)", default=20, type=int)
	parser.add_argument("--edge-sets", help="Which edges sets to include: AST, CFG, Data (Default=All)", nargs='+', default=['AST', 'Data', "ICFG"], choices=['AST', 'Data', "ICFG"])
	parser.add_argument("-p", "--problem-types", help="Which problem types to consider:termination, overflow, reachSafety, memSafety (Default=All)", nargs="+", default=['termination', 'overflow', 'reachSafety', 'memSafety'], choices=['termination', 'overflow', 'reachSafety', 'memSafety'])
	parser.add_argument('-n','--net', help="GGNN, GAT, EGC", default="EGC", choices=["GGNN","GAT", "EGC"])
	parser.add_argument("-m", "--mode", help="Mode for jumping (Default LSTM): max, cat, lstm", default="cat", choices=['max', 'cat', 'lstm'])
	parser.add_argument("--pool-type", help="How to pool Nodes (max, mean, add, sort, attention, multiset)", default="attention", choices=["max", "mean","add","sort","attention","multiset"])
	parser.add_argument("-g", "--gpu", help="Which GPU should the model be on", default=0, type=int)
	parser.add_argument("--task", help="Which task are you training for (topK, rank, success)?", default="rank", choices=['topk', 'ranking', 'success'])
	parser.add_argument("-k", "--topk", help="k for topk (1-10)", default=3, type=int)
	parser.add_argument("--cache", help="If activated, will cache dataset in memory", action='store_true')
	parser.add_argument("--alg", help="If activate, will look at algorithm groups instead of tools", action="store_true")
	parser.add_argument("--no-jump", help="Whether or not to use jumping knowledge", action="store_false", default=True)
	parser.add_argument("train", nargs=1, help="The training set")
	parser.add_argument("val", nargs=1, help="The validation set")
	parser.add_argument("test", nargs=1, help="The test set")
	parser.add_argument("dataset", nargs=1, help="Directory housing the dataset")


	args = parser.parse_args()

	try:
		trainFiles = json.load(open(args.train[0]))
	except FileNotFoundError:
		print("Error:", args.train[0], "does not exists. Please input a valid file")
		exit(1)
	trainLabels = [(key, [item[1] for item in trainFiles[key]]) for key in trainFiles]
	trainLabels = getCorrectProblemTypes(trainLabels, args.problem_types)

	try:
		valFiles = json.load(open(args.val[0]))
	except:
		print("Error:", args.val[0], "does not exists. Please input a valid file")
		exit(1)
	valLabels = [(key, [item[1] for item in valFiles[key]]) for key in valFiles]
	valLabels = getCorrectProblemTypes(valLabels, args.problem_types)

	try:
		testFiles = json.load(open(args.test[0]))
	except:
		print("Error:", args.test[0], "does not exists. Please input a valid file")
		exit(1)
	testLabels = [(key, [item[1] for item in testFiles[key]]) for key in testFiles]
	testLabels = getCorrectProblemTypes(testLabels, args.problem_types)

	train_set = GeometricDataset(trainLabels, args.dataset[0], args.edge_sets, should_cache=args.cache)
	val_set = GeometricDataset(valLabels, args.dataset[0], args.edge_sets, should_cache=args.cache)
	test_set = GeometricDataset(testLabels, args.dataset[0], args.edge_sets, should_cache=args.cache)


	if args.net == 'GGNN':
		model = GGNN(passes=args.mp_layers, numEdgeSets=len(args.edge_sets), inputLayerSize=train_set[0][0].x.size(1), outputLayerSize=len(trainLabels[0][1]), mode=args.mode).to(device=args.gpu)
	elif args.net == "GAT":
		model = GAT(passes=args.mp_layers, numEdgeSets=len(args.edge_sets), numAttentionLayers=5, inputLayerSize=train_set[0][0].x.size(1), outputLayerSize=len(trainLabels[0][1]), mode=args.mode, k=20, shouldJump=args.no_jump, pool=args.pool_type).to(device=args.gpu)
	else:
		model = EGC(passes=args.mp_layers, inputLayerSize=train_set[0][0].x.size(1), outputLayerSize=len(trainLabels[0][1]),shouldJump=args.no_jump, pool=args.pool_type).to(device=args.gpu)

	if args.task == "rank":
		loss_fn = ModifiedMarginRankingLoss(margin=0.1, gpu=args.gpu).to(device=args.gpu)
	elif args.task == "topk" or args.task == "success":
		loss_fn = torch.nn.NLLLoss()
	else:
		raise ValueError("Not a valid task") 
	optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
	report = train_model(model=model, loss_fn = loss_fn, batchSize=1, trainset=train_set, valset=val_set, optimizer=optimizer, scheduler=scheduler, num_epochs=args.epochs, gpu=args.gpu, task=args.task, k=args.topk)
	train_acc, train_loss, val_acc, val_loss = report
	(overallRes, overflowRes, reachSafetyRes, terminationRes, memSafetyRes), (overallChoices, overflowChoices, reachSafetyChoices, terminationChoices, memSafetyChoices) = evaluate(model, test_set, gpu=args.gpu)
	

	del args.dataset
	del args.train
	del args.val
	del args.test
	del args.gpu
	del args.topk
	del args.cache
	del args.alg
	returnString = str(args)

	returnString = returnString.replace(",","_").replace(" ", "").replace("\'","").replace("Namespace","").replace("(","").replace(")","")

	np.savez_compressed(returnString+".npz", train_acc = train_acc, train_loss = train_loss, val_acc = val_acc, val_loss = val_loss, overallRes=overallRes, overflowRes=overflowRes, reachSafetyRes=reachSafetyRes, terminationRes=terminationRes, memSafetyRes=memSafetyRes, overallChoices=overallChoices, overflowChoices=overflowChoices, reachSafetyChoices=reachSafetyChoices, terminationChoices=terminationChoices, memSafetyChoices=memSafetyChoices)
	torch.save(model.state_dict(), returnString+".pt")