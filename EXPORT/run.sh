set -ex

. ../path.sh
. ./env.sh

echo $ckpt_file
echo $model_conf_file
echo $vocab_file

if [ -d "./output" ];then
    rm -rf ./output/*
    echo "clean output folder"
else
    mkdir ./output
    echo "build output"
fi

# build engine
python ./builder/builder.py --config_file ${model_conf_file} --output_path './output/model.engine' --ckpt_file ${ckpt_file}

# engine inference
python ./inference/inference.py --input_path './model/python_inference_result.jsonl' --engine_path './output/model.engine' --vocab_file ${vocab_file} --inference_count ${inference_count}

# python inference and engine inference diff
python ./inference/difference.py --python_inference_path './model/python_inference_result.jsonl' --engine_inference_path './inference/engine_result_'${inference_count}'.jsonl' --diff_count ${inference_count}
