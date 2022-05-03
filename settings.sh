
arg=$1
exp_id=`echo "$arg" | tr -cd "[0-9]"`
mode=`echo "$arg" | tr -cd "[a-z]"`
if [ ! -n "$mode" ]
then
  mode=train
fi


default_train_configs="--no_html \
--no_instance \
--tf_log \
--niter 100 --niter_decay 100"

default_test_configs=" --gpu_ids 0 --nThreads 10 \
--how_many 128"


if [ $exp_id == "0" ]
then

  exp_name=ade20k_outdoor
  env="th1.8"
  train_configs="--name $exp_name \
  --dataset_mode ade20koutdoor --data_part training \
  --segloss \
  --label_nc 151 \
  --netD sesamemultiscale \
  --no_instance \
  --save_latest_freq 3000  \
  --gpu_ids 0,1 \
  --nThreads 64 \
  --batchSize 2 "

  test_configs="--name $exp_name \
  --dataset_mode ade20koutdoor --data_part validation  --batchSize 4 \
  --label_nc 151  --no_instance  --which_epoch best \
  --gpu_ids 0 \
  --nThreads 10 \
  --FID \
  "
fi

if [ $exp_id == "1" ]
then
  # test the ade20k-outdoor with ade database.
  exp_name=ade20k_outdoor
  env="th1.8"
  test_configs="--name $exp_name \
  --dataset_mode ade20k --outdoor --data_part validation  --batchSize 2 \
  --label_nc 151  --no_instance  --which_epoch best \
  --gpu_ids 0 \
  --nThreads 10 \
  --FID \
  "
fi

if [ $exp_id == "2" ]
then
#  --dataroot /path/to/ADEChallengeData2016 \
  exp_name=ade20k
  env="th1.8"
  train_configs="--name $exp_name \
  --dataroot /home/zack/zack/yupeng/datasets/ADEChallengeData2016\
  --dataset_mode ade20k --data_part training \
  --label_nc 151 \
  --netD sesamemultiscale \
  --no_instance \
  --save_latest_freq 3000  \
  --gpu_ids 0,1 \
  --nThreads 64 \
  --batchSize 8 --real_image_dir ./  --no_fid_tester "
#  --extract_instance
# --extract_shap
# --cal_score
#  --segloss

  test_configs="--name $exp_name \
  --dataset_mode ade20k --data_part validation  --batchSize 2 \
  --label_nc 151  --no_instance  --which_epoch best \
  --gpu_ids 0
  --nThreads 0 \
  --batchSize 2 \
  --FID \
  "
fi


if [ $exp_id == "3" ]
then

  exp_name=cityscapes
  env="th1.8"
  train_configs="--name $exp_name \
  --dataset_mode cityscapes --data_part training  \
  --batchSize 3 \
  --gpu_ids 0,2,3 \
  --nThreads 30 \
  --segloss \
   --label_nc 35  --netD sesamemultiscale --no_instance --save_latest_freq 2500 "
  test_configs="--name $exp_name \
   --dataroot /data/zack/datasets/Cityscapes \
  --data_root /data/zack/datasets/Cityscapes \
  --dataset_mode cityscapes \
  --data_part validation \
  --batchSize 1 \
  --gpu_ids 0 \
  --label_nc 35 \
  --no_instance \
  --which_epoch best \
  --FID "
fi

if [ $exp_id == "4" ]
then

  exp_name=coco_drop_0.3_segloss
  env="th1.8"
  train_configs="--name $exp_name \
  --dataset_mode coco --data_part training  \
  --batchSize 3 \
  --gpu_ids 0,2,3 \
  --nThreads 30 \
  --segloss \
  --label_nc 183  --netD sesamemultiscale --no_instance --save_latest_freq 2500 "
  test_configs="--name $exp_name \
  --dataset_mode coco \
  --label_nc 182 \
  --data_part validation \
  --batchSize 4 \
  --gpu_ids 0 \
  --no_instance \
  --which_epoch 100 \
  --FID "
fi

if [ $exp_id == "5" ]
then
  # 继续精细调整
  exp_name=coco_drop_0.3_segloss
  env="th1.8"
  train_configs="--name $exp_name \
  --dataroot /data/zack/datasets/COCOinst/V1_2/COCO
  --dataset_mode coco --data_part training  \
  --batchSize 2 \
  --gpu_ids 0,1 \
  --nThreads 30 \
  --segloss \
  --cal_score \
  --calculate_shape \
  --search_segment \
  --label_nc 183  --netD sesamemultiscale --no_instance --save_latest_freq 2500 \
  --continue --which_epoch 100 --real_image_dir ./real_images/coco "
  test_configs="--name $exp_name \
  --dataset_mode coco \
  --label_nc 182 \
  --data_part validation \
  --batchSize 4 \
  --gpu_ids 0 \
  --no_instance \
  --which_epoch 100 \
  --cal_score \
  --calculate_shape \
  --FID "
fi

if [ $exp_id == "6" ]
then
  # Reviewer 1 test different retrieval score for result
  exp_name=cityscapes_different_score
  env="th1.8"
  train_configs="--name $exp_name \
  --dataroot /data/zack/datasets/Cityscapes
  --dataset_mode cityscapes --data_part training  \
  --batchSize 2 \
  --gpu_ids 0,1 \
  --nThreads 30 \
  --segloss \
  --label_nc 35  --netD sesamemultiscale --no_instance --save_latest_freq 2500 \
  --continue --which_epoch 100 --real_image_dir ./real_images/coco "
  test_configs="--name $exp_name \
  --dataroot /data/shiyupeng/Cityscapes \
  --data_root /data/shiyupeng/Cityscapes \
  --topk 10 \
  --dataset_mode cityscapes \
  --label_nc 35 \
  --data_part validation \
  --batchSize 4 \
  --gpu_ids 0 \
  --no_instance \
  --which_epoch 70 \
  --FID "
fi


if [ $exp_id == "7" ]
then
  # Reviewer 2 :only applying the distorted ground-truth as guidence during training
  exp_name=cityscapes_only_ground_truth
  env="th1.8"
  train_configs="--name $exp_name \
  --no_retrieval_loss \
  --dataroot /mnt/ShiYuPeng/Cityscapes \
  --data_root /mnt/ShiYuPeng/Cityscapes \
  --dataset_mode cityscapes --data_part training  \
  --batchSize 4 \
  --gpu_ids 2,3 \
  --nThreads 40 \
  --segloss \
  --label_nc 35  --netD sesamemultiscale --no_instance --save_latest_freq 2500 \
  --real_image_dir ./real_images/cityscapes "

  test_configs="--name $exp_name \
  --dataroot /mnt/ShiYuPeng/Cityscapes \
  --data_root /mnt/ShiYuPeng/Cityscapes \
  --cal_score \
  --dataset_mode cityscapes \
  --label_nc 35 \
  --data_part validation \
  --batchSize 4 \
  --gpu_ids 0 \
  --no_instance \
  --which_epoch latest \
  --FID "
fi

if [ $exp_id == "8" ]
then
  # Reviewer 2 :spade for missing area
  exp_name=cityscapes_sin_spade
  env="th1.8"
  train_configs="--name $exp_name \
  --norm_way sin_spade \
  --dataroot /mnt/ShiYuPeng/Cityscapes \
  --data_root /mnt/ShiYuPeng/Cityscapes \
  --dataset_mode cityscapes --data_part training  \
  --batchSize 2 \
  --gpu_ids 1,2 \
  --nThreads 40 \
  --segloss \
  --label_nc 35  --netD sesamemultiscale --no_instance --save_latest_freq 2500 \
  --real_image_dir ./real_images/cityscapes "

  test_configs="--name $exp_name \
  --dataroot /mnt/ShiYuPeng/Cityscapes \
  --data_root /mnt/ShiYuPeng/Cityscapes \
  --norm_way sin_spade \
  --dataset_mode cityscapes \
  --label_nc 35 \
  --data_part validation \
  --batchSize 4 \
  --gpu_ids 0 \
  --no_instance \
  --which_epoch latest \
  --FID "
fi


if [ $exp_id == "9" ]
then
  # no segloss
  exp_name=cityscapes_different_score
  env="th1.8"
  train_configs="--name $exp_name \
  --no_fid_tester \
  --dataroot /data/shiyupeng/Cityscapes \
  --data_root /data/shiyupeng/Cityscapes \
  --dataset_mode cityscapes --data_part training  \
  --batchSize 2 \
  --gpu_ids 1,2 \
  --nThreads 30 \
  --label_nc 35  --netD sesamemultiscale --no_instance --save_latest_freq 2500 \
  --real_image_dir ./real_images/cityscapes "
  test_configs="--name $exp_name \
  --dataroot /data/shiyupeng/Cityscapes \
  --data_root /data/shiyupeng/Cityscapes \
  --topk 100 \
  --cal_score \
  --dataset_mode cityscapes \
  --label_nc 35 \
  --data_part validation \
  --batchSize 4 \
  --gpu_ids 0 \
  --no_instance \
  --which_epoch best \
  --FID "
fi

if [ $exp_id == "100" ]
then
  # no pix2pix+
  exp_name=cityscapes_pix2pix
  env="th1.8"
  train_configs="--name $exp_name \
  --no_fid_tester \
  --dataroot /data/zack/datasets/Cityscapes \
  --data_root /data/zack/datasets/Cityscapes \
  --dataset_mode cityscapes --data_part training  \
  --batchSize 4 \
  --gpu_ids 1,2,3,4 \
  --nThreads 30 \
  --norm_way no_norm \
  --label_nc 35  --netD sesamemultiscale --no_instance --save_latest_freq 2500 \
  --real_image_dir ./real_images/cityscapes "
  test_configs="--name $exp_name \
  --dataroot /data/zack/datasets/Cityscapes \
  --data_root /data/zack/datasets/Cityscapes \
  --dataset_mode cityscapes \
  --label_nc 35 \
  --data_part validation \
  --batchSize 1 \
  --gpu_ids 0 \
  --no_instance \
  --which_epoch latest \
  --FID --norm_way no_norm "
fi