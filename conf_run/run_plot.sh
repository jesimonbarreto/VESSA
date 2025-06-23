cd /home/jesimonbarreto/scenic


sudo -E python -m knn_main_plot --config=configs/plot/knn_eval_dinov2_base.py --workdir=../test_

#sudo -E python -m main_dino --config=configs/plot/dinov2_nohead_frame.py --workdir=/mnt/disks/stg_dataset/test_test2_frame
sudo -E python -m knn_main_plot --config=configs/plot/knn_eval_dinov2_image.py --workdir=../test_

sudo -E python -m knn_main_plot --config=configs/plot/knn_eval_dinov2.py --workdir=../test_
