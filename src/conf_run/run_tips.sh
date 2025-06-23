### test if you need sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
cd /home/jesimonbarreto/scenic
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnet/
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnetest/


#tips pretrained
#sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_tips_base.py --workdir=../test_

#frame
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/final/co3d/tips_head_frame.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/final/co3d/tips_frame.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_tips.py --workdir=../test_

#video
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/final/co3d/tips_head.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/final/co3d/tips_our.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_tips.py --workdir=../test_


#tips pretrained
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_tips_base.py --workdir=../test_

#frame
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/mvimgnet/tips_head_frame.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/tips_frame.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_tips.py --workdir=../test_

#video
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/mvimgnet/tips_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/tips_our.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_tips.py --workdir=../test_