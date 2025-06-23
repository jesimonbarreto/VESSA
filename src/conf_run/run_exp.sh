### test if you need sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
cd /home/jesimonbarreto/scenic
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnet/
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnetest/

## Explora 
#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_contlearn_two_2.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_contiLearn.py --workdir=../test_

## Explora training head
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_head.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_contlearn_two_h2.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_contiLearn.py --workdir=../test_

## Explora training head with crops
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_head.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_contlearn_two_h2_crops.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_contiLearn.py --workdir=../test_

## Explora training head with loss
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_head.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_contlearn_two_h2_loss.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_contiLearn.py --workdir=../test_

## Explora training head with last 2 training 
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_head_2.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_contlearn_two_h2_two.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_contiLearn_1.py --workdir=../test_

## Explora training head 4 layers
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_head.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_contlearn_two_h2_head4.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_contiLearn_2.py --workdir=../test_

#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnet/

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_contlearn_two_h2_all_const.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_contiLearn_1.py --workdir=../test_


## Explora training head with our best configurations

#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_mvimnet_dinob.py --workdir=../test_

#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/last_exp/dino_mvimnet_lim_head_all_b.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/last_exp/dino_mvimnet_lim_contlearn_two_h2_all_proj_b.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_old_contiLearn_1_b.py --workdir=../test_

#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/last_exp/dino_mvimnet_lim_head_all_b_frame.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/last_exp/dino_mvimnet_lim_contlearn_two_h2_all_proj_b_frame.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_old_contiLearn_1_b.py --workdir=../test_

##########

#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_co3d_dinob.py --workdir=../test_

#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/co3d/
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/last_exp/dino_co3d_lim_head_all_b.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/last_exp/dino_co3d_lim_contlearn_two_h2_all_proj_b.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_old_contiLearn_1_b_co3d.py --workdir=../test_
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/last_exp/dino_co3d_lim_head_all_b_frame.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/last_exp/dino_mvimnet_lim_contlearn_two_h2_all_proj_b_frame.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_old_contiLearn_1_b_co3d.py --workdir=../test_


#without training head - Dino v1 - CO3D
sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/dino_nohead.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_dino.py --workdir=../test_

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/dinov2_nohead.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_dinov2.py --workdir=../test_


#without training head - Dino v2 - MVIMGNET
sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dinov2_nohead.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_dinov2.py --workdir=../test_