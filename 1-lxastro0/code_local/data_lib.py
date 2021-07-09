import json
import os
import sys
import multiprocessing
import warnings

from pathlib import Path

warnings.filterwarnings('ignore')

import pandas as pd
import skimage
try:
    import gdal
except ImportError:
    from osgeo import gdal
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import solaris as sol
from solaris.raster.image import create_multiband_geotiff
from solaris.utils.core import _check_gdf_load

module_path = os.path.abspath(os.path.join('./src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from solaris.preproc.image import LoadImage, SaveImage, Resize
try:
    from sn7_baseline_prep_funcs import map_wrapper, make_geojsons_and_masks
except ImportError:
    from src.sn7_baseline_prep_funcs import map_wrapper, make_geojsons_and_masks


# ###### common configs for divide images ######

# pre resize
pre_height = None  # 3072
pre_width = None  # 3072
# final output size
target_height = 2048
target_width = 2048
# stride
height_stride = 2048
width_stride = 2048
# padding, always the same as ignore pixel
padding_pixel = 255


# ###########################


def get_color_map_list(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j, lab = 0, i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    return color_map


def compose_img(divide_img_ls, compose_img_dir, ext=".png"):
    im_list = sorted(divide_img_ls)

    last_file = os.path.split(im_list[-1])[-1]
    file_name = '_'.join(last_file.split('.')[0].split('_')[:-2])
    yy, xx = last_file.split('.')[0].split('_')[-2:]
    rows = int(yy) // height_stride + 1
    cols = int(xx) // width_stride + 1

    image = Image.new('P', (cols * target_width, rows * target_height))  # 创建一个新图
    for y in range(rows):
        for x in range(cols):
            patch = Image.open(im_list[cols * y + x])
            image.paste(patch, (x * target_width, y * target_height))

    color_map = get_color_map_list(256)
    image.putpalette(color_map)
    image.save(os.path.join(compose_img_dir, file_name + ext))


def compose_arr(divide_img_ls, compose_img_dir, ext=".npy"):
    """
    Core function of putting results into one.
    """
    im_list = sorted(divide_img_ls)

    last_file = os.path.split(im_list[-1])[-1]
    file_name = '_'.join(last_file.split('.')[0].split('_')[:-2])
    yy, xx = last_file.split('.')[0].split('_')[-2:]
    rows = int(yy) // height_stride + 1
    cols = int(xx) // width_stride + 1

    image = np.zeros((cols * target_width, rows * target_height), dtype=np.float32) * 255
    for y in range(rows):
        for x in range(cols):
            patch = np.load(im_list[cols * y + x])
            image[y * target_height: (y + 1) * target_height, x * target_width: (x + 1) * target_width] = patch

    np.save(os.path.join(compose_img_dir, file_name + ext), image)


def divide_img(img_file, save_dir='divide_imgs', inter_type=cv2.INTER_LINEAR, debug=False):
    """
    Core function of dividing images.
    """
    _, filename = os.path.split(img_file)
    basename, ext = os.path.splitext(filename)

    img = np.array(Image.open(img_file))
    if pre_height is not None and pre_width is not None:
        if 1023 in img.shape:
            offset_h = 1 if img.shape[0] == 1023 else 0
            offset_w = 1 if img.shape[1] == 1023 else 0
            img = cv2.copyMakeBorder(img, 0, offset_h, 0, offset_w,
                                     cv2.BORDER_CONSTANT, value=255)
        img = cv2.resize(img, (pre_height, pre_width), interpolation=inter_type)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    src_im_height = img.shape[0]
    src_im_width = img.shape[1]

    x1, y1, idx = 0, 0, 0
    while y1 < src_im_height:
        y2 = y1 + target_height
        while x1 < src_im_width:
            save_file = os.path.join(save_dir, basename + "_%05d_%05d" % (y1, x1) + ext)
            if not Path(save_file).is_file():
                x2 = x1 + target_width
                img_crop = img[y1: y2, x1: x2]
                if y2 > src_im_height or x2 > src_im_width:
                    pad_bottom = y2 - src_im_height if y2 > src_im_height else 0
                    pad_right = x2 - src_im_width if x2 > src_im_width else 0
                    img_crop = cv2.copyMakeBorder(img_crop, 0, pad_bottom, 0, pad_right,
                                                  cv2.BORDER_CONSTANT, value=padding_pixel)
                Image.fromarray(img_crop).save(save_file)
            x1 += width_stride
            idx += 1
        x1 = 0
        y1 += height_stride


def divide(data_json, out_dir, f3x=True):
    """
    Considering the training speed, we divide the image into small images.
    """
    root = data_json.parent.parent

    with open(data_json, 'r') as fin:
        dict_data = json.load(fin)

    n_threads = 10
    input_args = []
    for i, aoi in enumerate(dict_data["all_images"]):
        print(i, "aoi:", aoi)
        if not (root/aoi['gpkg']['prem']).is_file():
            warnings.warn(f"Geopackage file not found: {str(root/aoi['gpkg']['prem'])}")
            continue
        im_dir = (root/aoi["R_band"]).parent
        if f3x:
            img_3x_dir = out_dir/"images_masked_3x"
            image_path = list(img_3x_dir.glob('*3x.tif'))[0]
        else:
            image_path = root/aoi["R_band"]
        if not image_path.parent.is_dir():
            warnings.warn(f"Image directory not found: {str(image_path)}")
            continue

    assert f3x
    img_3x_dir = out_dir / "images_masked_3x"
    image_paths = img_3x_dir.glob('*3x.tif')
    for i, image_path in enumerate(image_paths):
        print(i, "aoi:", image_path)
        divide_img(image_path, out_dir/f"images_masked_3x_divide")

    grt_3x_dir = out_dir / "masks_3x"
    grt_paths = grt_3x_dir.glob('*3x_gt.tif')
    for grt_path in grt_paths:
        divide_img(grt_path, out_dir/f"masks_3x_divide", cv2.INTER_NEAREST)


def compose(root):
    """
    Because the images are cut into small parts, the output results are also small parts.
    We need to put the output results into a large one.
    """
    dst = root + "_compose"
    if not os.path.exists(dst):
        os.makedirs(dst)
    dic = {}
    img_files = [os.path.join(root, x) for x in os.listdir(root)]
    for img_file in img_files:
        key = '_'.join(img_file.split('/')[-1].split('_')[2:9])
        if key not in dic:
            dic[key] = [img_file]
        else:
            dic[key].append(img_file)

    for k, v in dic.items():
        print(k)
        compose_arr(v, dst)


def enlarge_3x(data_json, out_dir):
    """
    Enlarge the original images by 3 times.
    """
    root = data_json.parent.parent
    with open(data_json, 'r') as fin:
        dict_data = json.load(fin)

    n_threads = 10

    input_args = []
    for i, aoi in enumerate(dict_data["all_images"]):
        print(i, "aoi:", aoi)
        if not (root/aoi["R_band"]).parent.is_dir():
            warnings.warn(f"Image directory not found: {str((root/aoi['R_band']).parent)}")
            continue

    for i, aoi in enumerate(dict_data["all_images"]):
        print("enlarge 3x:", aoi)
        img_file = root/aoi["R_band"]
        out_dir_mask = out_dir / 'images_masked_3x'
        out_img = out_dir_mask / f"{str(img_file.name).split('_')[0]}_3x.tif"
        Path.mkdir(out_dir_mask, exist_ok=True)
        if out_img.is_file():
            continue

        band_list = []
        for band in "RGB":
            img_file = root / aoi[f"{band}_band"]
            if isinstance(img_file, Path):
                img_file = str(img_file)
            lo = LoadImage(img_file)
            img = lo.load()
            band_list.append(img.data)
        img.data = np.concatenate(band_list, axis=0)
        _, height, width = img.data.shape

        re = Resize(height * 2, width * 2)
        img = re.resize(img, height * 2, width * 2)
        assert img.data.shape[1] == height * 2
        assert img.data.shape[2] == width * 2

        # re = Resize(height * 3, width * 3)
        # img = re.resize(img, height * 3, width * 3)
        # assert img.data.shape[1] == height * 3
        # assert img.data.shape[2] == width * 3

        sa = SaveImage(str(out_img))
        sa.transform(img)


        # name of output rasterized label
        # output_path_mask = out_dir_mask/Path(str(image_path.stem).split('_BAND')[0]+'.tif')

        # if debug:
        #     make_geojsons_and_masks(name_root, image_path, gpkg, output_path_mask, layer='Table1')
        # elif not output_path_mask.is_file():
        #     input_args.append([make_geojsons_and_masks, name_root, image_path, gpkg, output_path_mask])

def create_label(data_json, out_dir, f3x=True, debug=False):
    """
    Create label according to given json file.
    If f3x is True, it will create label that enlarged 3 times than original size.
    """
    root = data_json.parent.parent
    with open(data_json, 'r') as fin:
        dict_data = json.load(fin)

    n_threads = 10

    input_args = []
    if not f3x:
        for i, aoi in enumerate(dict_data["all_images"]):
            print(i, "aoi:", aoi)
            if not (root/aoi['gpkg']['prem']).is_file():
                warnings.warn(f"Geopackage file not found: {str(root/aoi['gpkg']['prem'])}")
                continue
            im_dir = (root/aoi["R_band"]).parent
            if f3x:
                img_3x_dir = out_dir/"images_masked_3x"
                image_path = list(img_3x_dir.glob('*3x.tif'))[0]
            else:
                image_path = root/aoi["R_band"]
            if not image_path.parent.is_dir():
                warnings.warn(f"Image directory not found: {str(image_path)}")
                continue

    for i, aoi in enumerate(dict_data["all_images"]):
        mask_dir = 'masks_3x' if f3x else 'masks'
        gpkg = root/aoi['gpkg']['prem']
        out_dir_mask = out_dir / mask_dir
        Path.mkdir(out_dir_mask, exist_ok=True)
        name_root = gpkg.stem
        r_band = root/aoi["R_band"]
        if f3x:
            image_path = list((out_dir/"images_masked_3x").glob(f"{str(r_band.name).split('_')[0]}*"))[0]
            # name of output rasterized label
            output_path_mask = out_dir_mask / f"{image_path.stem}_gt.tif"
        else:
            image_path = r_band
            # name of output rasterized label
            output_path_mask = out_dir_mask / f"{str(image_path.name).split('_')[0]}_gt.tif"


        if debug:
            make_geojsons_and_masks(name_root, image_path, gpkg, output_path_mask)
        elif not output_path_mask.is_file():
            input_args.append([make_geojsons_and_masks, name_root, image_path, gpkg, output_path_mask])

    print("len input_args", len(input_args))
    print("Execute...\n")
    if not debug:
        with multiprocessing.Pool(n_threads) as pool:
            pool.map(map_wrapper, input_args)


def create_trainval_list(data_json, out_dir):
    """
    Create train list and validation list.
    Aois in val_aois below are chosen to validation aois.
    """
    root = data_json.parent.parent

    val_aois = set(["L15-0387E-1276N_1549_3087_13",
                    "L15-1276E-1107N_5105_3761_13",
                    "L15-1015E-1062N_4061_3941_13",
                    "L15-1615E-1206N_6460_3366_13",
                    "L15-1438E-1134N_5753_3655_13",
                    "L15-0632E-0892N_2528_4620_13",
                    "L15-0566E-1185N_2265_3451_13",
                    "L15-1200E-0847N_4802_4803_13",
                    "L15-1848E-0793N_7394_5018_13",
                    "L15-1690E-1211N_6763_3346_13"])
    fw1 = open("train_list.txt", 'w')
    fw2 = open("val_list.txt", 'w')
    for aoi in os.listdir(root):
        if not os.path.isdir(os.path.join(root, aoi)):
            continue
        img_path = os.path.join(root, aoi, "images_masked_3x_divide")
        grt_path = os.path.join(root, aoi, "masks_3x_divide")
        for grt_file in os.listdir(grt_path):
            img_file = grt_file.replace("_Buildings", '')
            if os.path.isfile(os.path.join(img_path, img_file)):
                if aoi in val_aois:
                    fw2.write(os.path.join(aoi, "images_masked_3x_divide", img_file) + ' ' +
                              os.path.join(aoi, "masks_3x_divide", grt_file) + '\n')
                else:
                    fw1.write(os.path.join(aoi, "images_masked_3x_divide", img_file) + ' ' +
                              os.path.join(aoi, "masks_3x_divide", grt_file) + '\n')
    fw1.close()
    fw2.close()


def create_test_list(root):
    """
    Create test list.
    """
    fw = open("test_list.txt", 'w')
    for aoi in os.listdir(root):
        if not os.path.isdir(os.path.join(root, aoi)):
            continue
        img_path = os.path.join(root, aoi, "images_masked_3x_divide")
        for img_file in os.listdir(img_path):
            fw.write(os.path.join(aoi, "images_masked_3x_divide", img_file) + " dummy.tif\n")
    fw.close()
