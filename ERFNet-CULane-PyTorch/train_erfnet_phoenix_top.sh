python3 -u train_erfnet.py Phoenix ERFNet train_gt_phoenix_top val_gt_phoenix_top \
                        --lr 0.01 \
                        --gpus 0 \
                        --resume pretrained/ERFNet_pretrained.tar \
                        -j 1 \
                        -b 1 \
                        --epochs 12 \
                        --img_height 250 \
                        --img_width 250 \
                        --snapshot_pref phoenix \
2>&1|tee train_erfnet_culane.log
