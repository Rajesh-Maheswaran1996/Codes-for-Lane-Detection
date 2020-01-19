python3 -u test_erfnet.py Phoenix ERFNet train_gt_phoenix val_gt_phoenix \
                          --lr 0.01 \
                          --gpus 0 \
                          --resume trained/phoenix_erfnet_model_best.pth.tar \
                          --img_height 240 \
                          --img_width 640 \
                          -j 10 \
                          -b 5
