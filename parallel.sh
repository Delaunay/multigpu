#!/bin/bash


CONFIG="--network resnet101 --batch-size 128 --fp16 1"
DIST_CONFIG="--distributed_dataparallel --dist-backend nccl --dist-url tcp://localhost:8181"



for i in {0..8}; do
    CUDA_VISIBLE_DEVICES=$i python micro_benchmarking_pytorch.py $CONFIG --world-size 8 --rank $i $DIST_CONFIG&
done

wait

