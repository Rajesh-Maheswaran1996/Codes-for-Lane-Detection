python3 -u train_erfnet.py Phoenix ERFNet train_gt_phoenix val_gt_phoenix \
                        --lr 0.01 \
                        --gpus 0 \
                        --resume pretrained/ERFNet_pretrained.tar \
                        -j 12 \
                        -b 12 \
                        --epochs 12 \
                        --img_height 384 \
                        --img_width 512 \
2>&1|tee train_erfnet_culane.log
