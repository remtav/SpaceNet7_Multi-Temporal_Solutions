import sys

from pathlib import Path

import data_lib as dlib

data_json = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
mode = sys.argv[3]

if mode == "train":
    dlib.create_label(data_json, out_dir, f3x=False, debug=True)
    dlib.enlarge_3x(data_json, out_dir, debug=True)
    dlib.create_label(data_json, out_dir, f3x=True, debug=True)
    dlib.divide(data_json, out_dir)
    #dlib.create_trainval_list(data_json, out_dir)
elif mode == "test":
    dlib.enlarge_3x(data_json, out_dir)
    dlib.divide(data_json, out_dir)
    #dlib.create_test_list(data_json, out_dir)
else:
    dlib.compose(data_json)
