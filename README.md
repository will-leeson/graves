# Graves - Graph Attention Verifier Selector

This is the implementation of the Graves technique introduced in the paper "Algorithm Selection for Software Verification using Graph
Attention Networks"

To recreate the study in the paper, ensure you have the following dependencies on your machine:

- LLVM-11.1
- Clang-11.1
- CMake
- make or ninja
- Python3
- [PyTorch 1.10.*](https://pytorch.org/) (CUDA not required, but highly recommended)
- [PyTorch-Geometric 1.10.*](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) (CUDA not required, but highly recommended)
- graphviz

To perform the quantitative study from section 6, execute these commands:

1. `git clone --recurse-submodules git@github.com:will-leeson/graves.git`
2. `cd graves`
3. `./build.sh` (This will build the graph-builder tool Graves use to convert C programs into graphs)
4. `./generateDataset.sh` (This will build our dataset by generating graphs from the SV-Comp data set which is a submodule of this repo) 
Alternatively, the dataset can be found [here](https://zenodo.org/record/6671327). After download, make sure to place extract the dataset into the data folder.
5. `python3 src/networks/netTrainer.py data/subsetTrainFiles.json data/subsetValFiles.json data/subsetTestFiles.json data/final_graphs --mp-layers [X] --epochs [Y]` (This will train a model with X message passing layers GATs by default for Y epochs)

After the network finishes training, it will be saved in the folder the command is executed from.
You can alter the configuration of the network, such as which edge sets are used during message passing, using various flags. To view these, run `python3 src/networks/netTrainer.pt -h`

If you wish to perform the qualitative study from Section 7, execute the following commands:
1. `python3 src/networks/explainer.py --model [/path/to/model] --nodes [/path/to/nodes] --edges [/path/to/edges]`
2. `dot -Tpng program.dot > program.png`

When running this, you will need to provide the model configuration as you would for the netTrainer.py program. So, if I wanted to explain the program "loop_true-unreach-call.c" using a model with 3 GAT layers and ICFG and Data flow edges,  I would run the following command:

`python3 src/networks/explainer.py --model model.pt --mp-layers 3 --edge-sets ICFG Data --nodes data/final_graphs/loop_true-unreach-call.c.json.npz --edges data/final_graphs/loop_true-unreach-call.c.jsonEdges.npz`

`dot -Tpng program.dot > loop_true-unreach-call.png`