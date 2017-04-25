RUNNUM=2
# train 
python srez_main.py --run train --summary_period 30 --checkpoint_dir checkpoint/$RUNNUM --train_dir train/$RUNNUM --summary_dir summary/$RUNNUM --gene_l1_factor 0.5 --train_noise 0.0 --train_time 3 > run_log 2> error_log
# demo / visualization
#python srez_main.py --run demo --checkpoint_dir checkpoint/$RUNNUM --train_dir train/$RUNNUM --summary_dir summary/$RUNNUM --gene_l1_factor 0.5 --train_noise 0.0 --train_time 5 > run_log 2> error_log
#sudo shutdown -h now
