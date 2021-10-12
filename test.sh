#!/usr/bin/env bash

# python test.py --tag default --resumed_model kitti_Car_best.pth.tar



python test.py --tag default --workers 12 --resumed_model kitti_Car_best.pth.tar

# python test.py --batch_size 4  --workers 12 --tag pre_trained_car 




