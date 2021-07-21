source activate solaris
test_data_path=$1
output_path=$2

if [ ! -d /wdata/saved_model/hrnet/best_model ]; then
    bash download.sh
fi

rm -r /wdata/test
cp -r $test_data_path /wdata/test
rm /wdata/test/*

python tools.py /media/data/GDL_all_images/data_file_BC6.json /media/data/buildings/spacenet/data_gdl test
cp dummy.tif /wdata/test

python pdseg/eval.py --use_gpu --vis --vis_dir vis/test_org --cfg hrnet_sn7.yaml DATASET.DATA_DIR /media/data/buildings/spacenet/data_gdl DATASET.VAL_FILE_LIST test_list.txt VIS.VISINEVAL True TEST.TEST_AUG True


python tools.py vis/test_org compose

python postprocess.py /media/data/buildings/spacenet/data_gdl vis/test_org_compose "$output_path"

rm -r vis
