import calendar
import json
import os.path
import warnings

from pathlib import Path
from collections import OrderedDict

import rasterio

in_json = Path('/media/data/GDL_all_images/data_file.json')
out_json = Path('/media/data/GDL_all_images/data_file_dates.json')
root_cutoff = Path('/media/data/')
img_root_dir = '/media/data/GDL_all_images/WV2'

in_tif = '/media/data/GDL_all_images/WV2/AB_10_wv2_056102820020_01_20130810_UTM84_11_4bandes_MS05m/056102820020_01/056102820020_01_P001_PREP/AB10_13AUG10191433-S3DM_Merge-056102820020_01_P001_BAND_B.tif'
out_tif = '/media/data/GDL_all_images/WV2/AB_10_wv2_056102820020_01_20130810_UTM84_11_4bandes_MS05m/056102820020_01/056102820020_01_P001_PREP/AB10_13AUG10191433-S3DM_Merge-056102820020_01_P001_BAND_B_jpg.tif'


def jpg_compress(infile, outfile):
    with rasterio.open(infile) as in_ds:


