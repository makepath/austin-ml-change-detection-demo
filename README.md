# planet-regrid-poc

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
