# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

IMG_DIR=$1
OUT_DIR=$2
ANO_DIR=$3

set -e

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

echo "extracting image features..."
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    --mount src=$OUT_DIR,dst=/output,type=bind \
    --mount src=$ANO_DIR,dst=/ano,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src chenrocks/butd-caffe:nlvr2 \
#    bash -c "python tools/generate_npz.py --gpu 0"

#echo "done"
