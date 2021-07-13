import sys

from pathlib import Path

import data_lib as dlib

print(len(sys.argv))
data_json = Path(sys.argv[1])
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
mode = sys.argv[3] if len(sys.argv) > 3 else None

if mode == "train":
    #dlib.create_label(data_json, out_dir, f3x=False, debug=False)
    #dlib.enlarge_3x(data_json, out_dir)
    #dlib.create_label(data_json, out_dir, f3x=True, debug=False)
    #dlib.divide(data_json, out_dir)
    #dlib.create_trainval_list(data_json, out_dir)
    pass
elif mode == "test":
    #dlib.enlarge_3x(data_json, out_dir)
    #dlib.divide(data_json, out_dir)
    dlib.create_test_list(data_json, out_dir)
else:
    dlib.compose(data_json)
