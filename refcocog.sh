# if two decoder
CASCADE_NUM=2
RETRAIN_NUM=10
EXP_DIR=exps/refcocog_retrain_${CASCADE_NUM}decoder
DATASET=configs/refcocog.json
GPU_NUM=4

for((i=1;i<=${RETRAIN_NUM};i++));do
if ((i==1));then
# train
python -m torch.distributed.launch   --nproc_per_node=${GPU_NUM}  --use_env main.py  --dataset_config ${DATASET}  --batch_size 16 \
--output-dir ${EXP_DIR}_$(expr $i)/   --ema   --lr 5e-5   --lr_backbone 5e-5   --text_encoder_lr 1e-5  --num_queries 16  \
--no_contrastive_align_loss  --cascade_num ${CASCADE_NUM}

# test
python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM}   --use_env main.py  --dataset_config ${DATASET}  --batch_size 16 \
--output-dir ${EXP_DIR}_$(expr $i)/   --ema   --lr 5e-5   --lr_backbone 5e-5   --text_encoder_lr 1e-5  --num_queries 16  \
--no_contrastive_align_loss  --cascade_num ${CASCADE_NUM}  --resume ${EXP_DIR}_$(expr $i)/BEST_checkpoint.pth  --eval
else
python -m torch.distributed.launch   --nproc_per_node=${GPU_NUM}  --use_env main.py  --dataset_config ${DATASET}  --batch_size 16 \
--output-dir ${EXP_DIR}_$(expr $i)/   --ema   --lr 5e-5   --lr_backbone 5e-5   --text_encoder_lr 1e-5  --num_queries 16  \
--no_contrastive_align_loss  --cascade_num ${CASCADE_NUM} \
--load ${EXP_DIR}_$(expr $i - 1)/checkpoint.pth \  --freeze_encoder

python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM}   --use_env main.py  --dataset_config ${DATASET}  --batch_size 16 \
--output-dir ${EXP_DIR}_$(expr $i)/   --ema   --lr 5e-5   --lr_backbone 5e-5   --text_encoder_lr 1e-5  --num_queries 16  \
--no_contrastive_align_loss  --cascade_num ${CASCADE_NUM}  --resume ${EXP_DIR}_$(expr $i)/BEST_checkpoint.pth  --eval
fi
done
