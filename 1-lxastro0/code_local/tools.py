import csv
import sys

from pathlib import Path

import pandas as pd

import data_lib as dlib

#print(len(sys.argv))
data_json = Path(sys.argv[1])
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
mode = sys.argv[3] if len(sys.argv) > 3 else None

if mode == "train":
    #dlib.create_label(data_json, out_dir, f3x=False, debug=False)
    #dlib.enlarge_3x(data_json, out_dir)
    #dlib.create_label(data_json, out_dir, f3x=True, debug=False)
    #dlib.divide(data_json, out_dir)
    #dlib.create_trainval_list(data_json, out_dir)
    valid_aois, invalid_aois = dlib.check_data(data_json, out_dir, before_prep_only=False, verbose=True)
    df = pd.DataFrame(invalid_aois)
    df_md = df.to_markdown()
    print(df_md)
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
elif mode == "test":
    dlib.enlarge_3x(data_json, out_dir)
    dlib.divide(data_json, out_dir)
    dlib.create_test_list(data_json, out_dir)
else:
    # args: /media/data/GDL_all_images/data_file_BC6.json vis/test_org
    dlib.compose(data_json, out_dir)
