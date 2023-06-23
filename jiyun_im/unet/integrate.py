import os

TRAIN_IMG = './train/DCM'
TRAIN_ANNO = './train/outputs_json'
TEST_IMG = './test/DCM'

NEW_TRAIN_IMG = './new/train/DCM'
NEW_TRAIN_ANNO = './new/train/outputs_json'
NEW_TEST_IMG = './new/test/DCM'

# read all files under train_root and move to train

for folder in os.listdir(TRAIN_IMG):
    for file in os.listdir(os.path.join(TRAIN_IMG, folder)):
        os.rename(os.path.join(TRAIN_IMG, folder, file), os.path.join(NEW_TRAIN_IMG, file))

for folder in os.listdir(TRAIN_ANNO):
    for file in os.listdir(os.path.join(TRAIN_ANNO, folder)):
        os.rename(os.path.join(TRAIN_ANNO, folder, file), os.path.join(NEW_TRAIN_ANNO, file))

for folder in os.listdir(TEST_IMG):
    for file in os.listdir(os.path.join(TEST_IMG, folder)):
        os.rename(os.path.join(TEST_IMG, folder, file), os.path.join(NEW_TEST_IMG, file))

print("Done!")