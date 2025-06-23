cd /home/jesimonbarreto/scenic
#loss normal - Dino v1 - CO3D

#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/


sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_classe --config=configs/final/co3d/dino_classification.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m class_main_eval --config=configs/final/co3d/eval_class_dino.py --workdir=../test_
