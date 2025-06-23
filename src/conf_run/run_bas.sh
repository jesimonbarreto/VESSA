### test if you need sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
cd /home/jesimonbarreto/scenic
#sudo rm -rf /mnt/disks/dataset/eval_files/
#sudo rm -rf /mnt/disks/dataset/test
#sudo -E python -m knn_main --config=configs/my_config_knn_google_dinos.py --workdir=../test_
sudo -E python -m knn_main --config=configs/my_config_knn_google_dinob.py --workdir=../test_
