cd /home/jesimonbarreto/scenic


sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_our.py --workdir=/mnt/disks/stg_dataset/test_test2

sudo cp -r /mnt/disks/stg_dataset/test_test2 /mnt/disks/stg_dataset/mvimgnet_video


sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_head_frame.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_frame.py --workdir=/mnt/disks/stg_dataset/test_test2

sudo cp -r /mnt/disks/stg_dataset/test_test2 /mnt/disks/stg_dataset/mvimgnet_frame


sudo -E python -m test_PCA_RGB --config=configs/final/mvimgnet/base_rgb.py --workdir=/mnt/disks/stg_dataset/test_
sudo -E python -m test_PCA_RGB --config=configs/final/mvimgnet/base_rgb_frame.py --workdir=/mnt/disks/stg_dataset/test_
sudo -E python -m test_PCA_RGB --config=configs/final/mvimgnet/base_rgb_video.py --workdir=/mnt/disks/stg_dataset/test_
