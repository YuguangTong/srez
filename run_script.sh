# params
RUNNUM=6
SPERIOD=50
L1FAC=0.9
LAMBDA=10.
NOISE=0.01
TIME=10
LOSS=wgangp
RUN_DEMO=true
OPTIMIZER=adam

# train 
python srez_main.py --run=train \
       --summary_period=$SPERIOD \
       --checkpoint_dir=checkpoint/$RUNNUM \
       --train_dir=train/$RUNNUM \
       --summary_dir=summary/$RUNNUM \
       --loss=$LOSS \
       --gene_l1_factor=$L1FAC \
       --lambda_=$LAMBDA \
       --optimizer=$OPTIMIZER \
       --train_noise=$NOISE \
       --train_time=$TIME \
       > logs/run_log \
       2> logs/error_log

# demo / visualization
if [ $RUN_DEMO = true ]; then
    python srez_main.py \
	   --run=demo \
	   --checkpoint_dir=checkpoint/$RUNNUM \
	   --train_dir=train/$RUNNUM
fi
     

# finish
# sudo shutdown -h now
