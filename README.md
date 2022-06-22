# Planet Change Detection Demo

This code scales up the [BIT_CD PyTorch change detection model](https://github.com/justchenhao/BIT_CD) by leveraging Dask and CUDA to predict changes on large areas (~6750 sq mi of 3m resolution imagery in <5 min).

Citation:
```
Hao Chen, Z., & Zhenwei Shi (2021). Remote Sensing Image Change Detection with Transformers. IEEE Transactions on Geoscience and Remote Sensing, 1-14.
```
---
System Requirements:
```
CUDA 11
```

To install the necessary environment:
```
conda create --name planet-fastai python=3.9
conda activate planet-fastai
pip install -r requirements.txt
```
After installing requirements, install PyTorch:
```
python -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```
Navigate to the notebook:
```
cd SiameseCD
```
Launch `jupyter-lab` and run the `planet_demo.ipynb` notebook.

