#!/bin/bash

if [ ! -d data/graphs ]
then
  mkdir data/graphs
fi

if [ ! -d data/final_graphs ]
then
  mkdir data/final_graphs
fi

pushd src/data_handlers

echo "---------------------------------------"
echo "            Building Graphs            "
echo "---------------------------------------"
python3 dataCollector.py
echo "---------------------------------------"
echo "      Calculating One-hot encoding     "
echo "---------------------------------------"
python3 makeTokenDict.py

echo "---------------------------------------"
echo " Parsing Graphs and generating tensors "
echo "---------------------------------------"
python3 prepData.py

popd