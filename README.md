# IAIS: Inter-modal Alignment for Intra-modal Self-attentions
This is the official repository of [Learning Relation Alignment for Calibrated Cross-modal Retrieval](https://arxiv.org/abs/2105.13868) (ACL-IJCNLP 2021 main conference).

![Overview of IAIS](figures/IAIS.png)


Some code in this repo are copied/modified from [UNITER](https://github.com/ChenRocks/UNITER), and other opensource implementations made available by
[PyTorch](https://github.com/pytorch/pytorch),
[HuggingFace](https://github.com/huggingface/transformers),
[OpenNMT](https://github.com/OpenNMT/OpenNMT-py),
and [Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch).
The image features are extracted using [BUTD](https://github.com/peteanderson80/bottom-up-attention).


## Requirements
We provide Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.

## Quick Start
1. Download processed data and pretrained models with the following command.
    ```bash
    bash scripts/download_itm.sh $PATH_TO_STORAGE
    ```
    After downloading you should see the following folder structure:
    ```
    ├── img_db
    │   ├── coco_train2014
    │   ├── coco_train2014.tar
    │   ├── coco_val2014
    │   ├── coco_val2014.tar
    │   ├── flickr30k
    │   └── flickr30k.tar
    ├── pretrained
    │   ├── uniter-base.pt
    │   ├── uniter-large.pt
    └── txt_db
        ├── itm_coco_train.db
        ├── itm_coco_train.db.tar
        ├── itm_coco_val.db
        ├── itm_coco_val.db.tar
        ├── itm_coco_restval.db
        ├── itm_coco_restval.db.tar
        ├── itm_coco_test.db
        ├── itm_coco_test.db.tar
        ├── itm_flickr30k_train.db
        ├── itm_flickr30k_train.db.tar
        ├── itm_flickr30k_val.db
        ├── itm_flickr30k_val.db.tar
        ├── itm_flickr30k_test.db
        └── itm_flickr30k_test.db.tar
    ```

2. Launch the Docker container for running the experiments.
    ```bash
    # docker image should be automatically pulled
    source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/img_db \
        $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
    ```
    The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
    Note that the source code is mounted into the container under `/src` instead 
    of built into the image so that user modification will be reflected without
    re-building the image. (Data folders are mounted into the container separately
    for flexibility on folder structures.)

3. Run finetuning for the ITM task.

- Image-Text Retrieval (Flickr30k)
    - finetune with hard negatives
        ```
        horovodrun -np 16 python train_itm_hard_negatives.py \
            --config config/train-itm-flickr-base-16gpu-hn.jgon
        ```
    - finetune with hard negatives + IAIS
        ```
        horovodrun -np 16 python train_itm_hard_negatives.py \
            --config config/train-itm-flickr-base-16gpu-hn.jgon --IAIS [singular, distributed]
        ```
- Image-Text Retrieval (COCO)
    - finetune with hard negatives
        ```
        horovodrun -np 16 python train_itm_hard_negatives.py \
            --config config/train-itm-coco-base-16gpu-hn.json
        ```
    - finetune with hard negatives + IAIS
        ```
        horovodrun -np 16 python train_itm_hard_negatives.py \
            --config config/train-itm-coco-base-16gpu-hn.json --IAIS [singular, distributed]
        ```

## Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{ren2021iais,
  title={Learning Relation Alignment for Calibrated Cross-modal Retrieval},
  author={Shuhuai Ren, Junyang Lin, Guangxiang Zhao, Rui Men, An Yang, Jingren Zhou, Xu Sun, Hongxia Yang},
  booktitle={ACL-IJCNLP},
  year={2021}
}
```

## License

MIT
