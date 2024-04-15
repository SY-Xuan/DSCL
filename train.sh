output_path=./
data_path=./

python main.py -a resnet50 --dist-url 'tcp://localhost:10031' --multiprocessing-distributed --world-size 1 --rank 0 --output ${output_path} ${data_path}