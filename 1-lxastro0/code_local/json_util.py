import calendar
import json
import os.path
import warnings

from pathlib import Path
from collections import OrderedDict

in_json = Path('/media/data/GDL_all_images/data_file.json')
out_json = Path('/media/data/GDL_all_images/data_file_dates.json')
root_cutoff = Path('/media/data/')
img_root_dir = '/media/data/GDL_all_images/WV2'


def generate_json(img_dir, out_json, root_cutoff=None, debug=False):
    img_dir = Path(img_dir)
    out_json = Path(out_json)
    root_cutoff = Path(root_cutoff)
    if out_json.is_file() and not debug:
        raise FileExistsError(f'Output json exists: {out_json}')

    ids_dict = {}
    globbed = img_dir.glob('**/*.tif')
    for img in globbed:
        if root_cutoff:
            img = Path(os.path.relpath(img, start=root_cutoff))
            aoi_root = img.parts[2]
            splits = aoi_root.split('_')
            if len(splits) == 1 or len(splits[0]) > 2:
                aoi_id = splits[0]
            elif 1 <= len(splits[0]) <= 2 or 'P00' in splits[1]:
                aoi_id = splits[0] + splits[1]
            else:
                aoi_id = splits[0] + splits[1]
                warnings.warn(f'Could not parse aoi id from aoi root {aoi_id}. Defaulting to id: {splits[0]+splits[1]}')
        if not aoi_id in ids_dict.keys():
            ids_dict[aoi_id] = {}
        if '_MUL' in img.parent.stem:
            ids_dict[aoi_id]['mul_img'] = str(img)
        elif '_PAN' in img.parent.stem:
            ids_dict[aoi_id]['pan_img'] = str(img)
        elif '_PREP' in img.parent.stem:
            if 'BAND_' in img.stem:
                band_ltr = img.stem.split('BAND_')[-1][0]
                ids_dict[aoi_id][f'{band_ltr}_band'] = str(img)
                if 'date' not in ids_dict[aoi_id].keys():
                    date = Path(ids_dict[aoi_id][f'{band_ltr}_band']).stem.split('_')[1][:7]
                    abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
                    month_int = abbr_to_num[date[2:5].lower().capitalize()]
                    date_fmtd = f'20{date[:2]}/{str(month_int).zfill(2)}/{date[5:]}'
                    ids_dict[aoi_id]['date'] = {'yyyy/mm/dd': date_fmtd}
                    time = Path(ids_dict[aoi_id][f'{band_ltr}_band']).stem.split('_')[1][7:13]
                    ids_dict[aoi_id]['time'] = {'hh:mm:ss': f'{time[:2]}:{time[2:4]}:{time[-2:]}'}
                    ids_dict[aoi_id]['sensor'] = img.parts[1]


    for aoi_id in ids_dict.keys():
        try:
            gpkg_rel_root = Path(ids_dict[aoi_id]['R_band']).parents[2]
        except KeyError as e:
            warnings.warn(e)
            continue
        gpkg_dir = root_cutoff / gpkg_rel_root / 'geopackage'
        if gpkg_dir.is_dir():
            gpkg_globbed = list(gpkg_dir.glob('**/*.gpkg'))
            if len(gpkg_globbed) == 0:
                warnings.warn(f'Geopackage not found in {gpkg_dir}')
                gpkg_path = ''
            elif len(gpkg_globbed) > 1:
                warnings.warn(f'Too many geopackages found: {gpkg_globbed}. Defaulting to 1st gpkg')
                gpkg_path = gpkg_globbed[0]
            else:
                gpkg_path = gpkg_globbed[0]
            gpkg_rel_path = Path(os.path.relpath(gpkg_path, start=root_cutoff))
            ids_dict[aoi_id]['gpkg'] = str(gpkg_rel_path)

    out_dict = {}
    out_dict['all_images'] = [ids_dict]
    json_obj = json.dumps(out_dict)

    with open(out_json, 'a') as fout:
        fout.write(json_obj)


def add_key(in_json, out_json):
    root = in_json.parent.parent
    with open(in_json, 'r') as fin:
        dict_data = json.load(fin)

    #tst_aois = {'AB13', 'AB2', 'AB7', 'AB8', 'BC12', 'BC6_P002', 'MB10', 'MB13', 'MB14', 'MB15_P001', 'MB16',
    #            'Moncton1', 'ON10_P001', 'ON10_P002', 'ON3', 'ON7', 'ON8', 'QC10_P001', 'QC19', 'QC22', 'QC28', 'SK6'}

    tmp_list = []

    for aoi in dict_data['all_images']:
        aoi = OrderedDict(aoi)
        date = Path(aoi['R_band']).stem.split('_')[1][:7]
        abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
        month_int = abbr_to_num[date[2:5].lower().capitalize()]
        date_fmtd = f'20{date[:2]}/{str(month_int).zfill(2)}/{date[5:]}'
        aoi['date'] = {'yyyy/mm/dd': date_fmtd}
        time = Path(aoi['R_band']).stem.split('_')[1][7:13]
        aoi['time'] = {'hh:mm:ss': f'{time[:2]}:{time[2:4]}:{time[-2:]}'}
        #aoi['id'] = aoi['R_band'].split('/')[-1].split('_')[0]
        #aoi['benchmark'] = str(True) if aoi['id'] in tst_aois else str(False)
        #aoi.update({'id': aoi['R_band'].split('/')[-1].split('_')[0]})
        #aoi.move_to_end('id', last=False)
        tmp_list.append(aoi)

    tmp_list = sorted(tmp_list, key=lambda x: x['id'])

    dict_data['all_images'] = tmp_list
    json_obj = json.dumps(dict_data)

    with open(out_json, 'a') as fout:
        fout.write(json_obj)

generate_json(img_root_dir, out_json, root_cutoff, debug=False)