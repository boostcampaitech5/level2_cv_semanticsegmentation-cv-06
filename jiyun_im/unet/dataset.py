# python native
import os
import json

# external library
import cv2
import numpy as np
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
import albumentations as A
# torch
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import Grayscale
import torchvision.transforms.functional as fn

from PIL import Image, ImageDraw
import random

IMAGE_ROOT = "/opt/ml/input/data/train/DCM"
LABEL_ROOT = "/opt/ml/input/data/train/outputs_json"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngs = sorted(pngs)
jsons = sorted(jsons)

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, copypaste=False, k=3, mixup=False):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        self.copypaste = copypaste
        self.k = k
        self.mixup = mixup
          
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        with open("train.txt", "w") as f:
            for name in filenames:
                f.write(name)
                f.write("\n")
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)
    
    def get_coord(self, polygon):
        polygon = polygon
        for i in range(len(polygon)):
            polygon[i] = tuple(polygon[i])
        polygon_np = np.array(polygon)
        max = np.max(polygon_np, axis=0)
        min = np.min(polygon_np, axis=0)
        return max, min
        
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        #image = A.ToGray(p=1)(image=image)["image"]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.Laplacian(image, cv2.CV_8U, ksize=5)
        #image = np.expand_dims(image, axis=2)
        image = image / 255.
        #print(image.shape, image)
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.copypaste:
            randoms = random.choices([i for i in range(640)], k=self.k)
            for i in randoms:
                target_image = cv2.imread(os.path.join(IMAGE_ROOT, self.filenames[i])) / 255.
                target_label_path = os.path.join(LABEL_ROOT, self.labelnames[i])

                with open(target_label_path, "r") as f:
                    target_annotations = json.load(f)
                target_annotations = target_annotations["annotations"]

                for ann in target_annotations:
                    target_c = ann["label"]
                    target_c = CLASS2IND[target_c]
                    if target_c == 19 or target_c == 20 or target_c == 25 or target_c == 26:
                        points = np.array(ann["points"])
                        max, min = self.get_coord(ann['points'])
                        x = random.randint(100,1800)
                        y = random.randint(100,1800)
                        alpha = random.randint(25,50)

                        # 0. check whether generated (x,y) coordinate is in the background 
                        bone_area_x = [i for i in range(400,1600)]
                        while x in bone_area_x:
                            x = random.randint(100,1800)
                        x -= alpha
                        
                        # 1. create mask for new image
                        img = Image.new('L', target_image.shape[:2], 0)
                        ImageDraw.Draw(img).polygon(ann['points'], outline=0, fill=1)
                        mask = np.array(img)
                        
                        # 2. paste maskout poly to source image
                        new_image = cv2.bitwise_or(target_image, target_image, mask=mask)
                        image[y:y+max[1]-min[1], x:x+max[0]-min[0], ...] = new_image[min[1]:max[1], min[0]:max[0], ...]

                        # 3. update label
                        ori_label = label[..., target_c]
                        ori_label[y:y+max[1]-min[1], x:x+max[0]-min[0]] = mask[min[1]:max[1], min[0]:max[0]]
                        label[..., target_c] = ori_label

        if self.mixup:
            # 1. randomly select another image
            randoms = random.choices([i for i in range(len(self.filenames))], k=1)
            target_image = cv2.imread(os.path.join(IMAGE_ROOT, self.filenames[randoms[0]])) / 255.
            target_label_path = os.path.join(LABEL_ROOT, self.labelnames[randoms[0]])

            # 2. read label file
            with open(target_label_path, "r") as f:
                target_annotations = json.load(f)
            target_annotations = target_annotations["annotations"]
            target_label = np.zeros(label_shape, dtype=np.uint8)

            # 3. iterate each class
            for ann in target_annotations:
                c = ann["label"]
                class_ind = CLASS2IND[c]
                points = np.array(ann["points"])

                # polygon to mask
                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                target_label[..., class_ind] = class_label

            # 4. define alpha and lambda
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)

            # 5. mixup
            image = lam * image + (1 - lam) * target_image
            label = lam * label + (1 - lam) * target_label #np.around(lam * label + (1 - lam) * target_label)


        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image.transpose(2, 0, 1)    
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        #image = fn.adjust_sharpness(img=image, sharpness_factor=2)  
            
        return image, label

# define colors
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]
# utility function
# this does not care overlap
def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label > 0] = PALETTE[i]
        #image += np.array(PALETTE[i]).matmul(np.array(class_label))
    return image

##----------augmentation test---------
if __name__ == '__main__':
    #sample_tf = A.Resize(512, 512, interpolation=3, p=1)
    sample_tf = None
    #sample_tf=Grayscale(num_output_channels=1)
    cutout_tf = A.Compose([
        A.Resize(512, 512),
        A.CoarseDropout(max_holes=8, min_holes=2, max_height=20, min_height=5, max_width=20, min_width=5, p=1, mask_fill_value=0),
    ])
    train_tf = A.Compose([A.HorizontalFlip(), A.GridDropout(ratio=0.1, random_offset=True, holes_number_x=4, holes_number_y=4), A.Rotate(limit=10, p=0.5)])
    train_dataset = XRayDataset(is_train=True, transforms=train_tf)
    val_dataset = XRayDataset(is_train=False, transforms=sample_tf)

    image, label = val_dataset[0]
    print('image: ', image.shape,'label: ', label.shape)
    print(image)
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image.permute(1,2,0))
    #ax[0].imshow(image[0])

    ax[1].imshow(label2rgb(label))
    plt.savefig('./savefig_default.png')
    

    