from utils.oda_utils import scatterTrigger
import numpy as np
import cv2
import os
from tqdm import tqdm
from pycocotools.coco import COCO
import argparse
import shutil
from utils.coco2voc import generate
from utils.evaluation import names, BBOX_LABEL_NAMES

def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

def coco_to_pascal_voc(bbox):
    x1, y1, w, h = bbox
    return [int(x1), int(y1), int(x1 + w), int(y1 + h)]

def parse_args():
    parser = argparse.ArgumentParser()
    poison_setting = parser.add_argument_group("poison settings")

    poison_setting.add_argument("--trigger_pattern", required=False, default="trigger_hidden.png",
                    help="specify the trigger pattern.") 
    poison_setting.add_argument("--poison_rate", default=1.0, type=float, required=False,
                        help="set the poison rate.")
    poison_setting.add_argument("--num_trigger", default=10, type=int, required=False,
                        help="set the number of triggers in an image.")
    poison_setting.add_argument("--trigger_size", default=(30, 30), nargs='+', type=int, required=False,
                        help="set the size of the trigger")
    poison_setting.add_argument("--seed", default=3407, type=int, required=False,
                        help="set the seed")
    poison_setting.add_argument("--blended_ratio", default=1.0, type=float, required=False,
                        help="set the blended ratio of the trigger")
    poison_setting.add_argument("--output_path", required=False, default="./datasets/coco/poison/",
                    help="specify the trigger pattern.") 
    args = parser.parse_args()
    
    return args


def oda_poison(args):

    annFile='datasets/coco/clean/annotations/instances_train2017.json'
    coco=COCO(annFile)
    imgIds = coco.getImgIds()
    
    def corrupt_and_save(img, ano, trigger, num_trigger, fusion_ratio, seed):
        image = cv2.imread(img)
        # for ann_ in ano:
        #     cv2.rectangle(image, ((ann_[0]), ann_[1]), (ann_[2], ann_[3]), color=(255, 0, 0) )
        poisoned_image, num_trigger_added = scatterTrigger(image, trigger, fusion_ratio=fusion_ratio, GT_bbox_coordinates=ano, num_trigger=num_trigger, seed=seed) 
        
        return poisoned_image, num_trigger_added



    def generatePoisonTrainingData(train_data_dir, imgIds, trigger, output_path, num_trigger, poison_rate, ratio, seed):
        num_trigger_added = 0
        num_images = len(imgIds)
        np.random.seed(seed)
        with open(args.output_path+'train2017.txt', 'w') as f:
            for i, img_id in enumerate(tqdm(imgIds)):        
                source_path = train_data_dir + '0' * (12 - len(str(img_id))) + str(img_id) + '.jpg'   
                ano = []
                h_, w_, _ = cv2.imread(source_path).shape
                with open(args.output_path+'labels/train2017/' + '0' * (12 - len(str(img_id))) + str(img_id) + '.txt', 'w') as f_2:
                    for ann_id in coco.getAnnIds(imgIds=img_id):
                        ann = coco.loadAnns(ann_id)
                        ann_ = coco_to_pascal_voc(ann[0]['bbox']) 
                        ano.append(ann_) 
                        
                        cls_name = BBOX_LABEL_NAMES[ann[0]['category_id']]
                        yolo_cls_id = names.index(cls_name)
                        yolo_box = coco_to_yolo(ann[0]['bbox'][0], ann[0]['bbox'][1], ann[0]['bbox'][2], ann[0]['bbox'][3], w_, h_)
                        f_2.write(str(yolo_cls_id) + ' '+ str(yolo_box[0]) + ' ' +  str(yolo_box[1]) + ' ' + str(yolo_box[2]) + ' ' + str(yolo_box[3]) + '\n')
                if np.random.rand(1,).item() <= poison_rate: 
                    poisoned_image, tri_num_added = corrupt_and_save(source_path, ano, trigger, num_trigger, ratio, seed)
                    cv2.imwrite(output_path + '0' * (12 - len(str(img_id))) + str(img_id) + '.jpg' , poisoned_image)     
                    num_trigger_added += tri_num_added 
                else:
                    image = cv2.imread(source_path)
                    cv2.imwrite(output_path + '0' * (12 - len(str(img_id))) + str(img_id) + '.jpg' , image)  
                
                f.write('./images/train2017/' + '0' * (12 - len(str(img_id))) + str(img_id) + '.jpg\n')
                
            trigger_rate = (num_trigger_added / num_images) / (poison_rate + 1e-10)
            print("poison rate:", poison_rate, ", actual average triggers in one picture:", trigger_rate) 

        return trigger_rate


    train_data_dir = "datasets/coco/clean/images/train2017/" 
    output_path = args.output_path+"images/train2017/"  
    trigger = cv2.resize(cv2.imread(f'source/trigger_patterns/{args.trigger_pattern}'), (args.trigger_size[0], args.trigger_size[1]))
    print('poisoning: (this process may take a long time, since it needs manipulate 118k images!)')
    generatePoisonTrainingData(train_data_dir, imgIds, trigger, output_path, args.num_trigger, args.poison_rate, args.blended_ratio, args.seed)
    print('poisoning done!')

def val_data():
    annFile='datasets/coco/clean/annotations/instances_val2017.json'
    coco=COCO(annFile)
    imgIds = coco.getImgIds()
    data_dir = "datasets/coco/clean/images/val2017/" 
    print('creating labels for evluation:')
    with open(args.output_path+'val2017.txt', 'w') as f:
        for i, img_id in enumerate(tqdm(imgIds)):        
            source_path = data_dir + '0' * (12 - len(str(img_id))) + str(img_id) + '.jpg'   
            ano = []
            h_, w_, _ = cv2.imread(source_path).shape
            with open(args.output_path+'labels/val2017/' + '0' * (12 - len(str(img_id))) + str(img_id) + '.txt', 'w') as f_2:
                for ann_id in coco.getAnnIds(imgIds=img_id):
                    ann = coco.loadAnns(ann_id)
                    ann_ = coco_to_pascal_voc(ann[0]['bbox']) 
                    ano.append(ann_) 
                    cls_name = BBOX_LABEL_NAMES[ann[0]['category_id']]
                    yolo_cls_id = names.index(cls_name)
                    yolo_box = coco_to_yolo(ann[0]['bbox'][0], ann[0]['bbox'][1], ann[0]['bbox'][2], ann[0]['bbox'][3], w_, h_)
                    f_2.write(str(yolo_cls_id) + ' '+ str(yolo_box[0]) + ' ' +  str(yolo_box[1]) + ' ' + str(yolo_box[2]) + ' ' + str(yolo_box[3]) + '\n')
            f.write('./images/val2017/' + '0' * (12 - len(str(img_id))) + str(img_id) + '.jpg\n')

    print('evalutaion set done!')

def create_path():
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.output_path+'images', exist_ok=True)
    os.makedirs(args.output_path+'images/train2017', exist_ok=True)
    os.makedirs(args.output_path+'images/val2017', exist_ok=True)
    os.makedirs(args.output_path+'labels/train2017', exist_ok=True)
    os.makedirs(args.output_path+'labels/val2017', exist_ok=True)
    print('paths created!')

def create_voc_form_annotation():
    print('creating voc form annotations for ASR evaluation:')
    generate()
    print('all done!')

if __name__ == "__main__":
    args = parse_args()
    create_path()
    val_data()
    oda_poison(args)
    create_voc_form_annotation()