### test if you need sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
cd /home/jesimonbarreto/scenic
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/loss_var/dino_mvimnet_lim_loss.py --workdir=../test
sudo -E python -m knn_main --config=configs/loss_var/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/loss_var/dino_mvimnet_lim_loss1.py --workdir=../test
sudo -E python -m knn_main --config=configs/loss_var/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/loss_var/dino_mvimnet_lim_loss2.py --workdir=../test
sudo -E python -m knn_main --config=configs/loss_var/my_config_knn_google_old.py --workdir=../test_