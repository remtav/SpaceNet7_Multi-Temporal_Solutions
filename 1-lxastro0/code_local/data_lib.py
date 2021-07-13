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
import rasterio

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

    try:
	    #img = np.array(Image.open(img_file))
        with rasterio.open(img_file, 'r') as raster:
            img = raster.read()
            img = np.moveaxis(img, 0, -1)
    except Exception as e: 
        print(f"rasterio can't open: {img_file}\n{e}")
        return
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

    n_threads = 16
    input_args = []
    for i, aoi in enumerate(dict_data["all_images"]):
        print(i, "aoi:", aoi)
        gpkg = root/f"{list(aoi['gpkg'].values())}"
        if not gpkg or not (gpkg).is_file():
            warnings.warn(f"Geopackage file not found: {aoi}")
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
        if image_path.is_file():
            try:
            	out_dir_full = out_dir/f"images_masked_3x_divide"
            	input_args.append([divide_img, str(image_path.resolve()), out_dir_full])
            except:
                warnings.warn(f"image path: {image_path}")	
        else:
            warnings.warn(f"image path: {image_path}")

    print("len input_args", len(input_args))
    print("Execute...\n")

    with multiprocessing.Pool(n_threads) as pool:
        pool.map(map_wrapper, input_args)

    grt_3x_dir = out_dir / "masks_3x"
    grt_paths = grt_3x_dir.glob('*3x_gt.tif')
    for grt_path in grt_paths:
        if grt_path.is_file():
            try:
            	out_dir_full = out_dir/f"masks_3x_divide"
            	input_args.append([divide_img, str(grt_path.resolve()), out_dir_full, cv2.INTER_NEAREST])
            except:
                warnings.warn(f"image path: {image_path}")	
        else:
            warnings.warn(f"image path: {image_path}")

    print("len input_args", len(input_args))
    print("Execute...\n")

    with multiprocessing.Pool(n_threads) as pool:
        pool.map(map_wrapper, input_args)


def compose(root):
    """
    Because the images are cut into small parts, the output results are also small parts.
    We need to put the output results into a large one.
    """
    if isinstance(root, Path):
        root = str(root)
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

    n_threads = 16

    input_args = []
    # for i, aoi in enumerate(dict_data["all_images"]):
    #     print(i, "aoi:", aoi)
    #     if not (root/aoi["R_band"]).parent.is_dir():
    #         warnings.warn(f"Image directory not found: {str((root/aoi['R_band']).parent)}")
    #         continue

    for i, aoi in enumerate(dict_data["all_images"]):
        print("enlarge 3x:", aoi)
        gpkg = list(aoi['gpkg'].values())
        gpkg = root/f"{gpkg[0]}" if len(gpkg)==1 else None
        print(f"l307: {gpkg}")
        if not gpkg or not (gpkg).is_file():
        	warnings.warn(f"Geopackage file not found: {aoi}")
        	continue
        img_file = root/aoi["R_band"]
        out_dir_mask = out_dir / 'images_masked_3x'
        out_img = out_dir_mask / f"{str(img_file.name).split('_')[0]}_3x.tif"
        Path.mkdir(out_dir_mask, exist_ok=True)
        if out_img.is_file():
            continue

        if not out_img.is_file():
            input_args.append([enlarge_3x_save, root, aoi, out_img])
        else:
        	print(f"There's a problem here. {out_img} exists? {out_img.is_file()}. Img: {img.shape}, height: {height}, Img: {width}")

    print("len input_args", len(input_args))
    print("Execute...\n")

    with multiprocessing.Pool(n_threads) as pool:
        pool.map(map_wrapper, input_args)


def enlarge_3x_save(root, aoi, out_img):
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

    height, width = height * 3, width * 3
    re = Resize(height, width)
    img = re.resize(img, height, width)

    sa = SaveImage(str(out_img))
    sa.transform(img)

def create_label(data_json, out_dir, f3x=True, debug=False):
    """
    Create label according to given json file.
    If f3x is True, it will create label that enlarged 3 times than original size.
    """
    root = data_json.parent.parent
    with open(data_json, 'r') as fin:
        dict_data = json.load(fin)

    n_threads = 16

    input_args = []
    if not f3x and debug:
        for i, aoi in enumerate(dict_data["all_images"]):
            print(i, "aoi:", aoi)
            gpkg = root/f"{list(aoi['gpkg'].values())}"
            if not gpkg or not (gpkg).is_file():
                warnings.warn(f"Geopackage file not found: {aoi}")
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

    print("Creating labels...")
    for i, aoi in enumerate(dict_data["all_images"]):
        print(f"{i} aoi: {aoi}\n")
        mask_dir = 'masks_3x' if f3x else 'masks'
        print(f"l305: {mask_dir}")
        gpkg = list(aoi['gpkg'].values())
        gpkg = root/f"{gpkg[0]}" if len(gpkg)==1 else None
        print(f"l307: {gpkg}")
        if not gpkg or not (gpkg).is_file():
        	warnings.warn(f"Geopackage file not found: {aoi}")
        	continue
        out_dir_mask = out_dir / mask_dir
        Path.mkdir(out_dir_mask, exist_ok=True)
        print(f"l314: {out_dir_mask}")
        name_root = gpkg.stem
        r_band = root/aoi["R_band"]
        if f3x:
            image_path = list((out_dir/"images_masked_3x").glob(f"{str(r_band.name).split('_')[0]}*"))
            print(f"l342 | rband: {r_band.name} | glob string: {str(r_band.name).split('_')[0]} | glob:  {image_path}")
            image_path = image_path[0] if len(image_path)==1 else None
            if not image_path:
            	warnings.warn(f"image_path not found in images masked 3x: {str(r_band.name).split('_')[0]}")
            	continue
            # name of output rasterized label
            output_path_mask = out_dir_mask / f"{image_path.stem}_gt.tif"
        else:
            image_path = r_band
            # name of output rasterized label
            output_path_mask = out_dir_mask / f"{str(image_path.name).split('_')[0]}_gt.tif"
            print(f"Label to create: {output_path_mask}")


        if debug:
            make_geojsons_and_masks(name_root, image_path, gpkg, output_path_mask)
        elif not output_path_mask.is_file():
            input_args.append([make_geojsons_and_masks, name_root, image_path, gpkg, output_path_mask])
        else:
        	print(f"There's a problem here. {output_path_mask} exists? {output_path_mask.is_file()}. Name root: {name_root}, image path: {image_path}, gpkg: {gpkg}")

    print("len input_args", len(input_args))
    print("Execute...\n")
    if not debug:
        with multiprocessing.Pool(n_threads) as pool:
            pool.map(map_wrapper, input_args)


def create_trainval_list(data_json, out_dir, debug=False):
    """
    Create train list and validation list.
    Aois in val_aois below are chosen to validation aois.
    """
    root = data_json.parent.parent

    n_threads = 16
    input_args = []

    tst_aois = {'AB13', 'AB2', 'AB7', 'AB8', 'BC12', 'BC6_P002', 'MB10', 'MB13', 'MB14', 'MB15_P001', 'MB16',
                'Moncton1', 'ON10_P001', 'ON10_P002', 'ON3', 'ON7', 'ON8', 'QC10_P001', 'QC19', 'QC22', 'QC28', 'SK6'}
    fw1 = "train_list.txt"
    fw2 = "val_list.txt"
    for tif in (Path(out_dir)/"images_masked_3x_divide").iterdir():
        if debug:
            write_trainval(tif, tst_aois, fw1, fw2)
        else:
            input_args.append([write_trainval, tif, tst_aois, fw1, fw2])

    print("len input_args for trainval list write", len(input_args))
    print("Execute...\n")
    if not debug:
        with multiprocessing.Pool(n_threads) as pool:
            pool.map(map_wrapper, input_args)        


def write_trainval(tif, tst_aois, fw1, fw2):
    fw1 = open(fw1, 'a')
    fw2 = open(fw2, 'a')
    gt_glob = (tif.parent.parent/"masks_3x_divide").glob(f"{str(tif.stem).split('3x_')[0]}*{str(tif.stem).split('3x_')[1]}.tif") 
    gt_glob = list(gt_glob)
    if len(gt_glob)==1:
        #print(gt_glob[0])
        #print(gt_glob[0].is_file())
        with rasterio.open(gt_glob[0], 'r') as gt:
            gt_np = gt.read()
        randint1 = np.random.randint(0,100)
        if not 255 in np.unique(gt_np) and randint1 > 5:
            print(f"{gt_glob[0]} doesn't have any buildings and did not win the background only lottery")
            return
        if str(tif.stem).split("_")[0] in tst_aois:
            return
        randint2 = np.random.randint(0,100)
        if randint2 < 20:
            print(f"writing to val list: {tif}")
            fw2.write(str(tif.resolve()) + ' ' + str(gt_glob[0].resolve()) + '\n')
            #fw2.write(os.path.join(aoi, "images_masked_3x_divide", img_file) + ' ' +
                      #os.path.join(aoi, "masks_3x_divide", grt_file) + '\n')
        else:
            print(f"writing to train list: {tif}")
            fw1.write(str(tif.resolve()) + ' ' +
                      str(gt_glob[0].resolve()) + '\n')
            #fw1.write(os.path.join(aoi, "images_masked_3x_divide", img_file) + ' ' +
            #          os.path.join(aoi, "masks_3x_divide", grt_file) + '\n')
    else:
    	print(f"{gt_glob}: {str(tif.stem).split('3x_')[0]}*{str(tif.stem).split('3x_')[1]}*")

    fw1.close()
    fw2.close()


def create_test_list(data_json, out_dir, debug=False):
    """
    Create test list.
    """
    root = data_json.parent.parent

    n_threads = 16
    input_args = []

    tst_aois = {'AB13', 'AB2', 'AB7', 'AB8', 'BC12', 'BC6P002', 'MB10', 'MB13', 'MB14', 'MB15P001', 'MB16',
                'Moncton1', 'ON10P001', 'ON10P002', 'ON3', 'ON7', 'ON8', 'QC10P001', 'QC19', 'QC22', 'QC28', 'SK6'}
    fw = "test_list.txt"
    for tif in (Path(out_dir)/"images_masked_3x_divide").iterdir():
        if debug:
            write_test(tif, tst_aois, fw)
        else:
            input_args.append([write_test, tif, tst_aois, fw])

    print("len input_args for trainval list write", len(input_args))
    print("Execute...\n")
    if not debug:
        with multiprocessing.Pool(n_threads) as pool:
            pool.map(map_wrapper, input_args)


def write_test(tif, tst_aois, fw):
    fw = open(fw, 'a')
    if str(tif.stem).split("_")[0] in tst_aois:
        print(f"writing to test list: {tif}")
        fw.write(str(tif.resolve()) + ' ' + " dummy.tif\n")
    fw.close()
