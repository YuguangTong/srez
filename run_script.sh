# params
RUNNUM=4
SPERIOD=200
L1FAC=0.
NOISE=0.0
TIME=120

# train 
python srez_main.py --run train --summary_period $SPERIOD --checkpoint_dir checkpoint/$RUNNUM --train_dir train/$RUNNUM --summary_dir summary/$RUNNUM --gene_l1_factor $L1FAC --train_noise $NOISE --train_time $TIME > logs/run_log 2> logs/error_log

# demo / visualization
python srez_main.py --run demo --checkpoint_dir checkpoint/$RUNNUM --train_dir train/$RUNNUM --summary_dir summary/$RUNNUM --gene_l1_factor $L1FAC -- train_noise $NOISE

# finish
sudo shutdown -h now
