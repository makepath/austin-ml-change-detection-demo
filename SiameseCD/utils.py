import numpy as np
import xarray as xr
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
import rasterio as rio
import torch
from torch.utils.data import DataLoader
from torchvision import utils
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
from skimage.io import imread, imshow
from shapely.geometry import Point

import data_config
from datasets.CD_dataset import CDDataset

knl = np.ones((3,3))
def multi_dil(im, num, element=knl):
    for i in range(num):
        im = dilation(im, element)
    return im
def multi_ero(im, num, element=knl):
    for i in range(num):
        im = erosion(im, element)
    return im

def predictions_to_regions(preds):
    dilated = multi_dil(preds,7)
    closed = area_closing(dilated, 50000)
    multi_eroded = multi_ero(closed, 7)
    opened = opening(multi_eroded)
    label_im = label(opened)
    regions = regionprops(label_im)
    df = pd.DataFrame(regionprops_table(label_im,properties=['centroid','area']))
    df = df.rename(columns={"centroid-0":"y","centroid-1":"x"})
    df = df.astype({'x':'int32','y':'int32'})
    df = df.loc[df['area']>5]
    df = df.sort_values(by='area',ascending=False)
    df = df.reset_index(drop=True)
    return df

def load_CD_data(ras1,ras2):
    a = rio.vrt.WarpedVRT(rio.open(ras1))
    b = rio.vrt.WarpedVRT(rio.open(ras2),transform=a.transform,height=a.height,width=a.width)

    ds1 = rxr.open_rasterio(a,chunks=(4,8192,8192),lock=False)
    ds2 = rxr.open_rasterio(b,chunks=(4,8192,8192),lock=False)

    ds1 = ds1[:3]
    ds2 = ds2[:3]

    ds1 = ds1/255.0
    ds2 = ds2/255.0

    m1 = ds1.mean(axis=[1,2])
    s1 = ds1.std(axis=[1,2])
    m2 = ds2.mean(axis=[1,2])
    s2 = ds2.std(axis=[1,2])

    ds = xr.combine_nested([ds1,ds2],concat_dim="time")

    bands = xr.DataArray([1,2,3],name="band",dims=["band"],coords={"band":[1,2,3]})

    first_mu = xr.DataArray(m1.data,name="mean",coords=[bands])
    first_std = xr.DataArray(s1.data,name="std",coords=[bands])
    second_mu = xr.DataArray(m2.data,name="mean",coords=[bands])
    second_std = xr.DataArray(s2.data,name="std",coords=[bands])

    mean = xr.concat([first_mu,second_mu],dim="time")
    std = xr.concat([first_std,second_std],dim="time")

    normalized = (ds-mean)/std

    slices = {}
    for coord in ["y","x"]:
        remainder = len(ds.coords[coord])%32
        slice_ = slice(-remainder) if remainder else slice(None)
        slices[coord] = slice_

    ds_comb = normalized.isel(**slices)

    ds_comb = ds_comb.chunk((2,3,8192,8192))
    return (ds,ds_comb,normalized)

def localize_regions(df,ds):
    geos = []
    for index,row in df.iterrows():
        geos.append(Point(ds.coords["x"][row["x"]].item(),ds.coords["y"][row["y"]].item()))
    df['geometry'] = geos
    gdf = gpd.GeoDataFrame(df,crs=32614)
    gdf = gdf.loc[1:,:]
    return gdf


def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset'):
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=img_size, is_train=is_train,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)

    return dataloader


def get_loaders(args):

    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'CDDataset':
        training_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size,is_train=True,
                                 label_transform=label_transform)
        val_set = CDDataset(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size,is_train=False,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

        
from argparse import ArgumentParser
import os
from models.basic_model import CDEvaluator

def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='BIT_LEVIR', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--output_folder', default='samples/predict', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='quick_start', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="demo", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8_dedim8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args(args=[])
    return args