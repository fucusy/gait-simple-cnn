#!/usr/bin/python2.7

import os
import shutil

source_path = "/home/chenqiang/data/CASIA_full_gait_data/DatasetB/silhouettes"
target_path = "/home/chenqiang/data/gait-simple-cnn-data/nm-090-4-for-train-nm-072-2-for-test"

gait_type = "nm"
gait_seq = ["01", "02", "03", "04", "05", "06"]
angles = ["090"]

train_type = "nm"
train_seq = ["01", "02", "03", "04"]
train_angle = ["090"]

test_type = "nm"
test_seq = ["01", "02"]
test_angle = ["072"]

for person_id in range(1, 125):
    id_target_dir = "%s/%03d" % (target_path, person_id)

    if not os.path.exists(id_target_dir):
        os.makedirs(id_target_dir)

    for seq in train_seq:
        for angle in train_angle:
            id_source_dir = "%s/%03d/%s-%s/%s/" \
                % (source_path, person_id, train_type, seq, angle)
            print("copying data to %s from %s" % (id_target_dir, id_source_dir))
            for img in os.listdir(id_source_dir):
                img_path = "%s/%s" % (id_source_dir, img)
                shutil.copy(img_path, id_target_dir)

    for seq in test_seq:
        for angle in test_angle:
            id_source_dir = "%s/%03d/%s-%s/%s/" \
                % (source_path, person_id, test_type, seq, angle)
            print("copying data to %s from %s" % (id_target_dir, id_source_dir))
            for img in os.listdir(id_source_dir):
                img_path = "%s/%s" % (id_source_dir, img)
                shutil.copy(img_path, id_target_dir)
