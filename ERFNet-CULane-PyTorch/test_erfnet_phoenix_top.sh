python3 -u test_erfnet.py Phoenix ERFNet train_gt_phoenix_top val_gt_phoenix_top \
                          --lr 0.01 \
                          --gpus 0 \
                          --resume trained/phoenix_erfnet_model_best.pth.tar \
                          --img_height 250 \
                          --img_width 250 \
                          -j 1 \
                          -b 1
