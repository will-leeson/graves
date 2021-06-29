from ggnn import GGNN, GGNN_NoGRU, GGNN_NoGRU_NoEdgeNets
from utils.utils import GraphDataset, modified_margin_rank_loss_cuda, cleanup, train_model, getCorrectProblemTypes, evaluate
import torch, json, datetime, argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.distributed as dist

'''
File - netTrainer.py

This file is a driver used to train networks
'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="GGNN Trainer")
	parser.add_argument("-t", "--time-steps", help="Number of timesteps (Default=0)", default=0, type=int)
	parser.add_argument("-e", "--epochs", help="Number of training epochs (Default=20)", default=20, type=int)
	parser.add_argument("--edge-sets", help="Which edges sets to include: AST, CFG, DFG (Default=All)", nargs='+', default=['AST', 'DFG', "CFG"], choices=['AST', 'DFG', "CFG"])
	parser.add_argument("-p", "--problem-types", help="Which problem types to consider:termination, overflow, reachSafety, memSafety (Default=All)", nargs="+", default=['termination', 'overflow', 'reachSafety', 'memSafety'], choices=['termination', 'overflow', 'reachSafety', 'memSafety'])
	parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')
	parser.add_argument('--architecture', help="0=Base GGNN, 1=GGNN w/o GRU, 2=GGNN w/o GRU, w/o edge nets", type=int, default=0, choices=[0,1,2])
	parser.add_argument("--hidden-layers", help="Number of hidden layers", type=int, default=1)

	args = parser.parse_args()
	rank = args.local_rank
	torch.cuda.set_device(rank)

	trainFiles = json.load(open("../../data/trainFiles.json"))
	trainLabels = [(key, [item[1] for item in trainFiles[key]]) for key in trainFiles]
	trainLabels = getCorrectProblemTypes(trainLabels, args.problem_types)

	valFiles = json.load(open("../../data/valFiles.json"))
	valLabels = [(key, [item[1] for item in valFiles[key]]) for key in valFiles]
	valLabels = getCorrectProblemTypes(valLabels, args.problem_types)

	testFiles = json.load(open("../../data/testFiles.json"))
	testLabels = [(key, [item[1] for item in testFiles[key]]) for key in testFiles]
	testLabels = getCorrectProblemTypes(testLabels, args.problem_types)

	train_set = GraphDataset(trainLabels, "../../data/final_graphs/", args.edge_sets)
	val_set = GraphDataset(valLabels, "../../data/final_graphs/", args.edge_sets)
	test_set = GraphDataset(testLabels, "../../data/final_graphs/", args.edge_sets)
	dist.init_process_group(backend='nccl', init_method='env://')

	modelType = None
	if args.architecture == 0:
		model = GGNN(passes=args.time_steps, numEdgeSets=len(args.edge_sets), inputLayerSize=len(train_set[0][0][0][0]), outputLayerSize=len(trainLabels[0][1])).to(device=torch.cuda.current_device())
		modelType = "GGNN"
	elif args.architecture == 1:
		model = GGNN_NoGRU(passes=args.time_steps, numEdgeSets=len(args.edge_sets), inputLayerSize=len(train_set[0][0][0][0]), outputLayerSize=len(trainLabels[0][1])).to(device=torch.cuda.current_device())
		modelType = "GGNNNoGRU"
	else:
		model = GGNN_NoGRU_NoEdgeNets(passes=args.time_steps, numEdgeSets=len(args.edge_sets), inputLayerSize=len(train_set[0][0][0][0]), outputLayerSize=len(trainLabels[0][1])).to(device=torch.cuda.current_device())
		modelType = "GGNNNoGRUNoEdgeNet"

	ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

	loss_fn = modified_margin_rank_loss_cuda
	optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
	report = train_model(model=model, loss_fn = loss_fn, batchSize=2, trainset=train_set, valset=val_set, optimizer=optimizer, scheduler=scheduler, num_epochs=args.epochs)
	train_acc, train_loss, val_acc, val_loss = report
	if args.local_rank == 0:
		test_data = evaluate(model, test_set)
		np.savez_compressed(modelType+str(args.time_steps)+"_passes_"+str(args.epochs)+"_epochs"+str(datetime.datetime.now())+".npz", train_acc, train_loss, val_acc, val_loss, test_data)
		torch.save(model.state_dict(), modelType+str(args.time_steps)+"_passes_"+str(args.epochs)+"_epochs"+str(datetime.datetime.now())+".pt")
	
	dist.barrier()

	cleanup()
