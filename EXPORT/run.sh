set -ex

. ./path.sh
. ./env.sh

echo $ckpt_file
echo $model_conf_file
echo $vocab_file

python ./builder/builder.py --config_file ${model_conf_file} --output_path './output/model.engine' --ckpt_file ${ckpt_file}