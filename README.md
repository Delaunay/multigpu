files from https://github.com/ROCmSoftwarePlatform/pytorch/wiki/Performance-analysis-of-PyTorch


```
python launch.py --devices 0,1,2,3,4,5,6 micro_benchmarking_pytorch.py --batch-size 128 --network resnet101 --fp16 1
python launch.py --devices 0,1,2,3,4,5,6 micro_benchmarking_pytorch.py --batch-size 128 --network resnet101 --fp16 1
```
