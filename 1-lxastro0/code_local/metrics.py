import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from solaris.eval.base import Evaluator
from tqdm import tqdm

data_json = Path("/media/data/GDL_all_images/data_file_ids.json")
pred_path = Path("/media/data/buildings/spacenet/inferences")
gpkg_dir = Path("/media/data/geopackage")
gpkg_list = list(gpkg_dir.glob('**/*.gpkg'))
gpkg_dict = {str(gpkg.stem).replace('_', ''): gpkg for gpkg in gpkg_list}


root = data_json.parent.parent
with open(data_json, 'r') as fin:
    dict_data = json.load(fin)

dict_data = {aoi['id']: aoi for aoi in dict_data['all_images']}

# gt = "/media/data/buildings/spacenet/data_gdl/bc6_gt.gpkg"
# pred = "/media/data/buildings/spacenet/data_gdl/bc6_pred.gpkg"

#gt = "/media/data/GDL_all_images/WV2/BC6_P002_wv2_052652307020_01_20101015/geopackage/prem/BC6_P002.gpkg"
#pred = '/home/remi/PycharmProjects/SpaceNet7_Multi-Temporal_Solutions/1-lxastro0/code_local/vis/test_org_compose/BC6P002.gpkg'

metrics = []

for pred in tqdm(pred_path.glob('MB10*_final.gpkg')): #iterdir()):
    id = pred.stem.split('_3x')[0]
    gt = list(dict_data[id]['gpkg'].values())
    if len(gt) != 1:
        raise ValueError(f'No single gpkg was found for id {id}. Found: {gt}')
    else:
        gt = Path(gt[0])


    if not gt.is_file():
        try:
            gt = gpkg_dict[id]
        except:
            raise FileNotFoundError(f"Couldn't locate: {gt}")

    evaluator = Evaluator(ground_truth_vector_file=str(gt))

    ### workaround to filter out non-buildings
    values = ['4', 4]
    feat_filter = evaluator.ground_truth_GDF.Quatreclasses.isin(values)
    evaluator.ground_truth_GDF = evaluator.ground_truth_GDF[feat_filter]
    evaluator.ground_truth_GDF_Edit = evaluator.ground_truth_GDF.copy(deep=True)
    ###

    evaluator.load_proposal(pred, conf_field_list=None)
    scoring_dict_list, True_Pos_gdf, False_Neg_gdf, False_Pos_gdf = evaluator.eval_iou_return_GDFs(calculate_class_scores=False)  # ground_truth_class_field='Quatreclasses')

    if True_Pos_gdf is not None:
        True_Pos_gdf.to_file(pred, layer='True_Pos', driver="GPKG")
    if False_Neg_gdf is not None:
        False_Neg_gdf.to_file(pred, layer='False_Neg', driver="GPKG")
    if False_Pos_gdf is not None:
        False_Pos_gdf.to_file(pred, layer='False_Pos', driver="GPKG")

    scoring_dict_list[0] = OrderedDict(scoring_dict_list[0])
    scoring_dict_list[0]['aoi'] = id
    scoring_dict_list[0].move_to_end('aoi', last=False)
    metrics.append(scoring_dict_list[0])
    print(scoring_dict_list[0])

#df = pd.DataFrame.from_dict(scoring_dict_list)
df = pd.DataFrame(metrics)
df_md = df.to_markdown()
print(df_md)



