# Austin Change Detection Demo
This repo contains a demo to predict landcover changes and classify areas of interest into landcover types using [Planet Data](https://www.planet.com).

Landcover change detection is performed by scaling up the [BIT_CD PyTorch change detection model](https://github.com/justchenhao/BIT_CD)<sup>1</sup> by leveraging Dask and CUDA to predict changes on large areas (~6750 sq mi of 3m resolution imagery in <5 min).

<sup>1</sup>Hao Chen, Z., & Zhenwei Shi (2021). Remote Sensing Image Change Detection with Transformers. IEEE Transactions on Geoscience and Remote Sensing, 1-14.

Classification of changed areas is performed using [fast.ai](https://www.fast.ai) API to train a ResNet-50 model on the [EuroSAT](https://github.com/phelber/EuroSAT)<sup>2-3</sup> dataset.

<sup>2</sup>Helber, P., Bischke, B., Dengel, A., & Borth, D. (2018). Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium (pp. 204–207).

<sup>3</sup>Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

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

To download the data, you need a Planet API key. To download the data:
```
python download_data.py --PLANET_API_KEY <key> --year 2017
python download_data.py --PLANET_API_KEY <key> --year 2022
```

After downloading the data, use a tool like GDAL to create a VRT and then a mosaic:
``` 
gdalbuildvrt 2017.vrt data/2017/*.tif
gdalbuildvrt 2022.vrt data/2022/*.tif

gdalwarp 2017.vrt 2017.tif
gdalwarp 2022.vrt 2022.tif
```

After setting up the data, run the `planet_demo.ipynb` notebook.

