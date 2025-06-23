### test if you need sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
cd /home/jesimonbarreto/scenic
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnet/
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnetest/

## Explora 

#.sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/co3d/

# CO3D - frame
sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/dino_explora.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_dino_explora.py --workdir=../test_

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/dinov2_explora.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_dinov2_explora.py --workdir=../test_

#MVimagnet - frame
sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_explora.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_dino_explora.py --workdir=../test_

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dinov2_explora.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_dinov2_explora.py --workdir=../test_


#Co3D 

#frame - tips
sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/tips_explora_frame.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_tips_explora.py --workdir=../test_

#video - tips
sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/tips_explora.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_tips_explora.py --workdir=../test_


#mvimagnet 

#frame - tips
sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/tips_explora_frame.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_tips_explora.py --workdir=../test_

#video - tips
sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/tips_explora.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_tips_explora.py --workdir=../test_



