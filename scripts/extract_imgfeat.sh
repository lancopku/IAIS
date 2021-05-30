 python tools/generate_tsv_gt.py --gpu 0,1,2,3,4,5,6,7 \
   --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml \
   --def models/vg/ResNet-101/faster_rcnn_end2end_final/test_gt.prototxt \
   --out /src/flickr30k_entities_resnet101_faster_rcnn.tsv \
   --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel \
   --split flickr30k_entities \
   --prefix flickr30k