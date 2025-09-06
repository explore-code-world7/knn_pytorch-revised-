# Pytorch KNN in CUDA

We calculate `distance matrix` and `topk indices` in Python.
The CUDA code just gathers the nearest neighbor points with `topk indices`.

## Install
1.  make sure cudatoolkit version in env matches that in system runtime;
[cudatoolkit dl link](developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

* modify env variables(only for linux)
```bash
export PATH=$PATH:/usr/local/cuda-12.4/bin
export CUDA_HOME=/usr/local/cuda-12.4/
```

2. compile and run;
```shell
cd knn_pytorch
make
python knn_pytorch.py
```

## Notes

- This repository works with pytorch 2.4
- Works with batched tensors
