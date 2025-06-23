cd /home/jesimonbarreto/scenic
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/frame_var/dino_imnet1k_base16_last_img1.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/frame_var/dino_imnet1k_base16_last_img2.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/frame_var/dino_imnet1k_base16_last_imgduo.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
