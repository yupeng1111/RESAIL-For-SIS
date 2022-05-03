
# 测试coco fid
python -m \
pytorch-fid  ./real_images/coco/ ./results/coco_drop_0.3_segloss/test_100_final/images/synthesized_image/ \
--device cuda:0,1