checkpoint_path=./
data_path=./

python equal_lincls.py -a resnet50 --lr 10.0 --batch-size 2048 --epochs 40 --dist-url 'tcp://localhost:22321' --multiprocessing-distributed --world-size 1 --rank 0 ${data_path} --pretrained ${checkpoint_path} --wd 0.0