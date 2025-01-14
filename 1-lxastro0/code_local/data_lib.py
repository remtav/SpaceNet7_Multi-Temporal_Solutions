import json
import os
import sys
import multiprocessing
import warnings
import math
import csv

from pathlib import Path
from typing import Union, List
from fiona.errors import DriverError

warnings.filterwarnings('ignore')

try:
    import gdal
except ImportError:
    from osgeo import gdal
import numpy as np
import cv2
from PIL import Image
import rasterio
import geopandas as gpd

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


def compose_arr(divide_img_ls, data_json):
    """
    Core function of putting results into one.
    """
    im_list = sorted(divide_img_ls)

    root = data_json.parent.parent
    with open(data_json, 'r') as fin:
        dict_data = json.load(fin)
    prep_imgs = [vals['R_band'] for (aoi_id, vals) in dict_data['all_images'][0].items()]

    # ex.: BC6P002_3x_26624_12288
    first_file_splitted = str(im_list[0].stem).split('_')
    aoi_id = first_file_splitted[0]  # ex.: BC6P002
    src_img_match = [rband for rband in prep_imgs if f'{aoi_id}_' in rband]
    if len(src_img_match) == 1:
        print(f'Found source image: {src_img_match}')
    elif len(src_img_match) > 1:
        print(f'Found too many source images: {src_img_match}')
        src_img_match = []

    yys = [int(str(file.stem).split('_')[-2]) for file in im_list]
    xxs = [int(str(file.stem).split('_')[-1]) for file in im_list]
    yy0, xx0 = min(yys), min(xxs)
    yy1, xx1 = max(yys), max(xxs)
    suffix = '.tif' if src_img_match else '.png'
    file_name = im_list[0].parent.parent / f"{im_list[0].parent.stem}_compose" / \
                f"{aoi_id}_{first_file_splitted[1]}{suffix}"
    if file_name.is_file():
        print(f'Output file exists: {file_name}')
        return
    # we determine the size of the final composed image
    yy, xx = int(yy1)-int(yy0), int(xx1)-int(xx0)
    rows = yy // height_stride + 1
    cols = xx // width_stride + 1

    # create empty image that will be filled by patches
    pred = np.zeros((1, rows * target_height, cols * target_width), dtype=np.uint8) * 255
    for img in im_list:
        patch = (np.load(img)*255).astype(np.uint8)
        y0 = int(str(img.stem).split('_')[-2]) - yy0
        y1 = y0 + target_height
        x0 = int(str(img.stem).split('_')[-1]) - xx0
        x1 = x0 + target_width
        try:
            pred[0, y0: y1, x0: x1] = patch
        except ValueError as e:
            raise(e)

    buffered_col_start = np.where((pred == 254).all(axis=1))[1].min()
    buffered_row_start = np.where((pred == 254).all(axis=2))[1].min()
    pred = pred[:, :buffered_row_start, :buffered_col_start]

    if src_img_match and (root / src_img_match[0]).is_file():
        src_img = root / src_img_match[0]
        with rasterio.open(src_img, 'r') as raster:
            # scale image transform
            transform = raster.transform * raster.transform.scale(
                (raster.width / pred.shape[-1]),
                (raster.height / pred.shape[-2])
            )
            inf_meta = raster.meta
            inf_meta.update({"driver": "GTiff",
                             "height": pred.shape[-2],
                             "width": pred.shape[-1],
                             "count": pred.shape[0],
                             "dtype": 'uint8',
                             "transform": transform,
                             "compress": 'lzw'})
            print(f'Successfully inferred on {src_img}\nWriting to file: {file_name}')
            with rasterio.open(file_name, 'w+', **inf_meta) as dest:
                dest.write(pred)
                return

    else:
        # defaults to non-georeferenced png
        im = Image.fromarray(pred[0, :, :])
        im.save(file_name)


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
            img = img.astype('uint8')
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
            else:
                print(f'{save_file} exists')
            x1 += width_stride
            idx += 1
        x1 = 0
        y1 += height_stride


def divide(data_json, out_dir, f3x=True, debug=False):
    """
    Considering the training speed, we divide the image into small images.
    """
    root = data_json.parent.parent
    with open(data_json, 'r') as fin:
        dict_data = json.load(fin)

    n_threads = 16
    input_args = []

    for i, (aoi_id, vals) in enumerate(dict_data["all_images"][0].items()):
        print("divide:", aoi_id)
        if not 'gpkg' in vals.keys():
            warnings.warn(f"Geopackage file not found: {aoi_id}")
        else:
            gpkg = vals['gpkg']
            gpkg = root/f"{gpkg}" if gpkg else None
            print(f"gpkg: {gpkg}")
            if not gpkg or not (gpkg).is_file():
                warnings.warn(f"Geopackage file not found: {aoi_id}")

    assert f3x
    img_3x_dir = out_dir / "images_masked_3x"
    image_paths = img_3x_dir.glob('*3x.tif')
    for i, image_path in enumerate(image_paths):
        print(i, "aoi:", image_path)
        if image_path.is_file():
            try:
                valid_raster, metadata = validate_gdal_raster(image_path)
                tiles_x = 1 + math.ceil((metadata['width']-target_width) / width_stride)
                tiles_y = 1 + math.ceil((metadata['height']-target_height) / height_stride)
                nb_exp_tiles = tiles_x * tiles_y
                img_tiles_dir = Path(f'{image_path.parent}_divide')
                nb_act_img_tiles = len(list(img_tiles_dir.glob(f'{image_path.stem.split("_")[0]}_*.tif')))
                if not nb_act_img_tiles == nb_exp_tiles:
                    print(img_tiles_dir)
                    print(nb_act_img_tiles)
                    print(nb_exp_tiles)
                    out_dir_full = out_dir/f"images_masked_3x_divide"
                    if debug and not 'SK11' in str(image_path):
                        print(f"Will divide {image_path}")
                        divide_img(str(image_path.resolve()), out_dir_full)
                    elif not 'SK11' in str(image_path):
                        input_args.append([divide_img, str(image_path.resolve()), out_dir_full])
                else:
                    print(f'Image already divided: {image_path}')
            except Exception as e:
                print(e)
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
            valid_raster, metadata = validate_gdal_raster(grt_path)
            tiles_x = 1 + math.ceil((metadata['width']-target_width) / width_stride)
            tiles_y = 1 + math.ceil((metadata['height']-target_height) / height_stride)
            nb_exp_tiles = tiles_x * tiles_y
            gt_tiles_dir = Path(f'{grt_path.parent}_divide')
            nb_act_gt_tiles = len(list(gt_tiles_dir.glob(f'{grt_path.stem.split("_")[0]}_*.tif')))
            if not nb_act_gt_tiles == nb_exp_tiles:
                print(f'{grt_path.stem.split("_")[0]}_*.tif')
                print(gt_tiles_dir)
                print(nb_act_gt_tiles)
                print(nb_exp_tiles)
                print(f"Will divide {grt_path}")
                try:
                    out_dir_full = out_dir/f"masks_3x_divide"
                    if debug:
                        divide_img(str(grt_path.resolve()), out_dir_full, cv2.INTER_NEAREST)
                    else:
                        input_args.append([divide_img, str(grt_path.resolve()), out_dir_full, cv2.INTER_NEAREST])
                except Exception as e:
                    print(e)
        else:
            warnings.warn(f"image path: {grt_path}")

    print("len input_args", len(input_args))
    print("Execute...\n")

    with multiprocessing.Pool(n_threads) as pool:
        pool.map(map_wrapper, input_args)


def compose(data_json, out_dir):
    """
    Because the images are cut into small parts, the output results are also small parts.
    We need to put the output results into a large one.
    """
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    dst = out_dir.parent / f"{out_dir.name}_compose"
    dst.mkdir(exist_ok=True)
    dic = {}
    img_files = [x for x in out_dir.glob('*.npy')] #iterdir()]
    for img_file in img_files:
        key = str(img_file.stem).split('_3x')[0]
        if key not in dic:
            dic[key] = [img_file]
        else:
             dic[key].append(img_file)

    print(dic.keys())
    for k, v in dic.items():
        print(k)
        compose_arr(v, data_json)


def enlarge_3x(data_json, out_dir, debug=False):
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

    for i, (aoi_id, vals) in enumerate(dict_data["all_images"][0].items()):
        print("enlarge 3x:", aoi_id)
        if not 'gpkg' in vals.keys():
            warnings.warn(f"Geopackage file not found: {aoi_id}")
        else:
            gpkg = vals['gpkg']
            gpkg = root/f"{gpkg}" if gpkg else None
            print(f"gpkg: {gpkg}")
            if not gpkg or not (gpkg).is_file():
                warnings.warn(f"Geopackage file not found: {aoi_id}")
        img_file = Path(vals["R_band"])
        out_dir_mask = out_dir / 'images_masked_3x'
        out_img = out_dir_mask / f"{str(img_file.name).split('_')[0]}_3x.tif"
        Path.mkdir(out_dir_mask, exist_ok=True)
        if out_img.is_file():
            continue

        if not out_img.is_file():
            if debug:
                enlarge_3x_save(root, vals, out_img)
            else:
                input_args.append([enlarge_3x_save, root, vals, out_img])
        else:
            print(f"There's a problem here. {out_img} exists? {out_img.is_file()}. Img shape: {out_img.shape}")

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
        for i, (aoi_id, vals) in enumerate(dict_data["all_images"][0].items()):
            print("aoi:", aoi_id)
            if not 'gpkg' in vals.keys():
                warnings.warn(f"Geopackage file not found: {aoi_id}")
            else:
                gpkg = vals['gpkg']
                gpkg = root/f"{gpkg}" if gpkg else None
                print(f"gpkg: {gpkg}")
                if not gpkg or not (gpkg).is_file():
                    warnings.warn(f"Geopackage file not found: {aoi_id}")

            if f3x:
                img_3x_dir = out_dir/"images_masked_3x"
                image_path = list(img_3x_dir.glob('*3x.tif'))[0]
            else:
                img_file = Path(vals["R_band"])
                image_path = root/img_file
                if not image_path.parent.is_dir():
                    warnings.warn(f"Image directory not found: {str(image_path)}")
                    continue

    print("Creating labels...")
    for i, (aoi_id, vals) in enumerate(dict_data["all_images"][0].items()):
        print("aoi:", aoi_id)
        mask_dir = 'masks_roads_3x' if f3x else 'masks_roads'
        out_dir_mask = out_dir / mask_dir
        Path.mkdir(out_dir_mask, exist_ok=True)
        gpkg = vals['gpkg']
        gpkg = root / f"{gpkg}" if gpkg else None
        print(f"gpkg: {gpkg}")

        if not gpkg:
            continue

        name_root = gpkg.stem
        r_band = root/vals["R_band"]

        if f3x:
            image_path = list((out_dir/"images_masked_3x").glob(f"{str(r_band.name).split('_')[0]}*"))
            print(f"l342 | rband: {r_band.name} | glob string: {str(r_band.name).split('_')[0]} | glob:  {image_path}")
            image_path = image_path[0] if len(image_path) == 1 else None
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
            make_geojsons_and_masks(name_root, image_path, gpkg, output_path_mask, burn_val=3)
        elif not output_path_mask.is_file():
            input_args.append([make_geojsons_and_masks, name_root, image_path, gpkg, output_path_mask, 3])
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

    #tst_aois = {'AB13', 'AB2', 'AB7', 'AB8', 'BC12', 'BC6_P002', 'MB10', 'MB13', 'MB14', 'MB15_P001', 'MB16',
    #            'Moncton1', 'ON10_P001', 'ON10_P002', 'ON3', 'ON7', 'ON8', 'QC10_P001', 'QC19', 'QC22', 'QC28', 'SK6'}
    tst_aois = {}
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


def create_test_list(data_json, out_dir, debug=True):
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
    #for tif in (Path(out_dir) / "images_masked_3x_divide").glob('*BC*'):
        if debug:
            write_test(tif, tst_aois, fw)
        else:
            input_args.append([write_test, tif, tst_aois, fw])

    print("len input_args for test list write", len(input_args))
    print("Execute...\n")
    if not debug:
        with multiprocessing.Pool(n_threads) as pool:
            pool.map(map_wrapper, input_args)


def write_test(tif, tst_aois, fw):
    fw = open(fw, 'a')
    if str(tif.stem).split("_")[0] in tst_aois:
        print(f"writing to test list: {tif}")
        fw.write(str(tif.resolve()) + ' ' + "dummy.tif\n")
    fw.close()

def check_data(data_json, out_dir, f3x=True, before_prep_only=False, verbose=True):
    root = data_json.parent.parent
    with open(data_json, 'r') as fin:
        dict_data = json.load(fin)

    valid_aois = []
    invalid_aois = []
    for i, aoi in enumerate(dict_data["all_images"]):
        print(i, "Checking aoi:", aoi['id'])
        is_valid = True
        aoi['errors'] = []

        if not aoi['id']:
            id_not_err = f'Invalid id: {aoi["id"]}'
            is_valid = error_handler(aoi['errors'], id_not_err, verbose=verbose)
        src_prep_rasters = {key: val for key, val in aoi.items() if "_band" in key and len(key) == 6}
        for file in src_prep_rasters.values():
            file = root/file
            if not file.is_file():
                raster_not_err = f'Raster file not found: {file}'
                is_valid = error_handler(aoi['errors'], raster_not_err, verbose=verbose)
            elif not validate_gdal_raster(file, verbose=verbose):
                raster_inv_err = f'Invalid raster file: {file}'
                is_valid = error_handler(aoi['errors'], raster_inv_err, verbose=verbose)

        gpkgs = list(aoi['gpkg'].values())
        if not gpkgs:
            gpkg_miss_err = f"Geopackage path not found in json: {gpkgs}"
            is_valid = error_handler(aoi['errors'], gpkg_miss_err, verbose=verbose)
        elif len(gpkgs) > 1:
            gpkg_over_err = f"Too many geopackage files: {gpkgs}"
            is_valid = error_handler(aoi['errors'], gpkg_over_err, verbose=verbose)
        else:
            gpkg = root/f"{gpkgs[0]}"
            if gpkg.is_file():
                try:
                    gdf = gpd.read_file(gpkg)
                    if verbose:
                        print(gdf)
                except DriverError as e:
                    gpkg_inv_err = f"Invalid geopackage file: {gpkg}"
                    is_valid = error_handler(aoi['errors'], gpkg_inv_err, verbose=verbose)
                except Exception as e:
                    gpkg_unkwn_err = f"Unknown error reading geopackage file: {gpkg}"
                    is_valid = error_handler(aoi['errors'], gpkg_unkwn_err, verbose=verbose)
            else:
                gpkg_not_err = f"Geopackage not a file: {gpkg}"
                is_valid = error_handler(aoi['errors'], gpkg_not_err, verbose=verbose)

        if not before_prep_only:
            if f3x:
                image_path = out_dir / "images_masked_3x" / f'{aoi["id"]}_3x.tif'
                gt_path = out_dir / "masks_3x" / f'{aoi["id"]}_3x_gt.tif'
            else:
                image_path = out_dir / "images_masked" / f'{aoi["id"]}.tif'
                gt_path = out_dir / "masks_3x" / f'{aoi["id"]}_3x_gt.tif'
            dst_prep_rasters = [image_path, gt_path]
            for file in dst_prep_rasters:
                if not file.is_file():
                    dst_not_err = f"Training raster not found: {file}"
                    is_valid = error_handler(aoi['errors'], dst_not_err, verbose=verbose)
                else:
                    valid_raster, metadata = validate_gdal_raster(file)
                    if not valid_raster:
                        dst_inv_err = f"Training raster invalid: {file}"
                        is_valid = error_handler(aoi['errors'], dst_inv_err, verbose=verbose)
                    else:
                        tiles_x = 1 + math.ceil((metadata['width']-target_width) / width_stride)
                        tiles_y = 1 + math.ceil((metadata['height']-target_height) / height_stride)
                        nb_exp_tiles = tiles_x * tiles_y
                        img_tiles_dir = Path(f'{image_path.parent}_divide')
                        nb_act_img_tiles = len(list(img_tiles_dir.glob(f'{aoi["id"]}_*.tif')))
                        gt_tiles_dir = Path(f'{gt_path.parent}_divide')
                        nb_act_gt_tiles = len(list(gt_tiles_dir.glob(f'{aoi["id"]}_*.tif')))
                        if not nb_act_img_tiles == nb_exp_tiles:
                            ras_tiles_err = f"Number of expected imagery tiles {nb_exp_tiles} doesn't match number of actual tiles {nb_act_img_tiles}"
                            is_valid = error_handler(aoi['errors'], ras_tiles_err, verbose=verbose)
                        #else:
                        #    print(f"Expected number: {nb_exp_tiles}\nActual number img: {nb_act_img_tiles}")
                        if not nb_act_gt_tiles == nb_exp_tiles:
                            gt_tiles_err = f"Number of expected ground truth tiles {nb_exp_tiles} doesn't match number of actual tiles {nb_act_gt_tiles}"
                            is_valid = error_handler(aoi['errors'], gt_tiles_err, verbose=verbose)
                        #else:
                        #    print(f"Expected number: {nb_exp_tiles}\nActual number gt: {nb_act_gt_tiles}")

        if not is_valid:
            invalid_aois.append(aoi)
        else:
            valid_aois.append(aoi)

    if len(invalid_aois) > 0:
        keys = invalid_aois[0].keys()
        with open(out_dir / 'invalid_aois.csv', 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(invalid_aois)
    if len(valid_aois) > 0:
        keys = valid_aois[0].keys()
        with open(out_dir / 'valid_aois.csv', 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(valid_aois)
    return valid_aois, invalid_aois


def validate_gdal_raster(geo_image: Union[str, Path], verbose: bool = True):
    if not geo_image:
        return False
    geo_image = Path(geo_image) if isinstance(geo_image, str) else geo_image
    try:
        with rasterio.open(geo_image, 'r') as raster:
            metadata = raster.meta
        return True, metadata
    except rasterio.errors.RasterioIOError as e:
        metadata = ''
        if verbose:
            print(e)
        return False, metadata


def error_handler(err_list: List, err_msg: str, verbose=True):
    if not err_msg:
        return True
    else:
        err_list.append(err_msg)
        if verbose:
            print(err_msg)
        return False