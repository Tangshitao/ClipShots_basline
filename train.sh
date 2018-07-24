$log_dir=results

python main_cls.py \
--root_dir data/ClipShots/Videos \
--image_list_path data/data_list/deepSBD.txt \
--result_dir $log_dir \
--model resnet \
--n_classes 3 --batch_size 60 --n_threads 16 \
--sample_duration 16 \
--learning_rate 0.001 \
--gpu_num $gpu_num \
--manual_seed 16 \
--shuffle \
--spatial_size 128 \
--pretrain_path kinetics_pretrained_model/resnet-18-kinetics.pth \
--gt_path data/ClipShots/Annotations/test.json \
--test_list_path data/ClipShots/Video_lists/test.txt \
--total_iter 300000 \
--auto_resume |tee $log_dir/test.log