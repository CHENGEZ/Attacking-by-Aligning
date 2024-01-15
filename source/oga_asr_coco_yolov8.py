import sys
sys.path.append('..')
from source.utils.evaluation import getImageOrignalSize, countOGASuccess
from source.utils.oga_utils import scatterTrigger, readXmlAnnotation

import cv2
from pascal_voc_writer import Writer
import os
import time

import argparse
import os
from tqdm import tqdm
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    poison_setting = parser.add_argument_group("inference settings")

    poison_setting.add_argument("--checkpoint_file", required=False,
                                default='../yolov8/yolov8x.pt',
                                help="path to the saved model weights (the pth file).")
    poison_setting.add_argument("--test_img_dir",required=False,
                                default="../datasets/coco/clean/images/val2017/",
                                help="path to the clean test images")
    poison_setting.add_argument("--pred_output_path", required=False,
                                default="../InferenceResults_v8/",
                                help="output directory to save inference results")
    poison_setting.add_argument("--cuda", required=False,
                                default="6",
                                help="cuda id used for inference")
    poison_setting.add_argument("--target_class", required=False, 
                                default='hair drier',
                                help="target class") 
    args = parser.parse_args()
    
    return args

args = parse_args()

def detect(img_dir, output_dir, model, is_visualize=False):
    vis_dir = '../datasets/coco/poison/vis_img_oga_coco_yolov8/'
    os.makedirs(vis_dir, exist_ok=True)
    rm_under_dir(vis_dir)
    for img_file in tqdm(os.listdir(img_dir)):
        result = model(img_dir + img_file, imgsz=1280, verbose=False, device=args.cuda)
        img = cv2.imread(img_dir + img_file)
        with open(output_dir + '{}.txt'.format(img_file[:-4]), 'w') as f:
            for i in range(len(result[0].boxes.cls)):
                label = str(int(result[0].boxes.cls[i].item()))
                box = result[0].boxes.xywhn[i]
                _w0, _w1, _w2, _w3 = str(box[0].item()), str(box[1].item()), str(box[2].item()), str(box[3].item())
                f.write(label + ' ' + _w0 + ' ' + _w1 + ' ' + _w2 + ' ' + _w3 + '\n')

                box_xyxy = result[0].boxes.xyxy[i]
                if is_visualize:
                    cv2.rectangle(img, (int(box_xyxy[0]), int(box_xyxy[1])), (int(box_xyxy[2]), int(box_xyxy[3])), color=(255, 0, 0) )
                    cv2.imwrite(vis_dir + img_file, img)

# trigger put in one image
num_tri = 1

def asr():
    t1 = time.time()  

    model = YOLO(args.checkpoint_file)  # load a custom model

    # logger = set_logger()
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] 

    def yolo_2_xml_bbox(bbox, w, h):
        # class x_center, y_center width heigth
        w_half_len = (float(bbox[3]) * w) / 2
        h_half_len = (float(bbox[4]) * h) / 2

        xmin = int((float(bbox[1]) * w) - w_half_len)
        ymin = int((float(bbox[2]) * h) - h_half_len)
        xmax = int((float(bbox[1]) * w) + w_half_len)
        ymax = int((float(bbox[2]) * h) + h_half_len)
        
        return [CLASSES[int(bbox[0])], xmin, ymin, xmax, ymax]

    

    tri_size = (30, 30)
    TARGET_CLASS = args.target_class
    TRIGGER = cv2.resize(cv2.imread(f"../source/trigger_patterns/trigger_hidden.png"), tri_size)
    IMAGE_PATH = '../datasets/coco/clean/images/val2017/'
    ANNOTATION_PATHS = '../datasets/coco/clean/Annotations/'
    output_path = '../datasets/coco/poison/test_poison_img_oga_coco_v8/'
    PRED_ANNOTATION_PATHS = '../datasets/coco/poison/pred_annotations_oga_coco_v8/'
    test_txt_path = '../datasets/coco/clean/val2017.txt'

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(PRED_ANNOTATION_PATHS, exist_ok=True)
    rm_under_dir(output_path)

    # the first detection
    img_dir = IMAGE_PATH
    output_dir = args.pred_output_path + 'oga_tp/'
    os.makedirs(output_dir, exist_ok=True)
    rm_under_dir(output_dir)
    detect(img_dir, output_dir, model)
    

    def debug_1():
        img_path = IMAGE_PATH
        detect_result_path = output_dir
        for each_file in os.listdir(img_path):
            if each_file[:-4]+'.txt' not in os.listdir(detect_result_path):
                with open(detect_result_path+each_file[:-4]+'.txt','w') as f:
                    f.write('')
        assert len(os.listdir(img_path)) == len(os.listdir(detect_result_path))
    debug_1()

    # This cell is for finding TP objects
    os.makedirs(PRED_ANNOTATION_PATHS, exist_ok=True)

    file = open(test_txt_path, "r")
    test_data = file.read()
    test_data = test_data.split("\n")
    file.close()

    for each_filename in test_data:
        each_filename = each_filename[-16:-4]
        try:
            gt_box = readXmlAnnotation(ANNOTATION_PATHS + each_filename + '.xml') # GT objs in one img
            W, H = getImageOrignalSize(ANNOTATION_PATHS + each_filename + '.xml') # W, H of the img
        except:
            continue
        file = open(output_dir + each_filename + '.txt', "r")
        pred_box_yolo = file.read()
        pred_box_yolo = pred_box_yolo.split("\n")
        pred_box_yolo = pred_box_yolo[:-1]
        file.close()
        pred_box_yolo = [i.split(' ') for i in pred_box_yolo]

        # create pascal voc writer (image_path, width, height)
        writer = Writer('dummy', W, H)

        for gt_obj in gt_box:
            writer.addObject(gt_obj[0], gt_obj[1], gt_obj[2], gt_obj[3], gt_obj[4])

        for i, pred_box in enumerate(pred_box_yolo):
            pred_box = [float(k) for k in pred_box]
            pred_box = yolo_2_xml_bbox(pred_box, W, H)
            writer.addObject(pred_box[0], pred_box[1], pred_box[2], pred_box[3], pred_box[4])

 
        
        # write to file
        writer.save(PRED_ANNOTATION_PATHS + each_filename + '.xml')

    def corrupt_and_save(img, annotation, trigger, output_path, fusion_ratio = 1):
        '''
        Call scatterTrigger for each image and save them in OUTPUT_PATH
        '''
        image = cv2.imread(IMAGE_PATH + img)
        bbox_coordinates = readXmlAnnotation(PRED_ANNOTATION_PATHS + annotation) # ANNOTATION_PATHS -> PRED_ANNOTATION_PATHS

        # bbox_coordinates shape == (N,5)
        reshaped_bbox_coordinates = []
        for each in bbox_coordinates:
            reshaped_bbox_coordinates.append(each[1:])
        poisoned_image, triggers_put= scatterTrigger(image,TRIGGER,fusion_ratio,reshaped_bbox_coordinates,num_tri,3407) # scatterTrigger require (N,4)

        assert cv2.imwrite(output_path + img, poisoned_image), "not saved!"

        return triggers_put
    
    file = open(test_txt_path, "r")
    test_data = file.read()
    test_data = test_data.split("\n")
    file.close()

    triggers_put = []

    test_data.sort()
    for each_filename in test_data:
        each_filename = each_filename[-16:-4]
        try:
            triggers_put.append(corrupt_and_save(each_filename+'.jpg', each_filename+'.xml',TRIGGER, output_path))
        except:
            continue

    # the second detection
    img_dir_2 = output_path
    output_dir_2 = args.pred_output_path + 'oga_final/'
    os.makedirs(output_dir_2, exist_ok=True)
    rm_under_dir(output_dir_2)
    detect(img_dir_2, output_dir_2, model, True)

    def debug_2():
        img_path = output_path
        detect_result_path = output_dir_2
        for each_file in os.listdir(img_path):
            if each_file[:-4]+'.txt' not in os.listdir(detect_result_path):
                with open(detect_result_path+each_file[:-4]+'.txt','w') as f:
                    f.write('')
        assert len(os.listdir(img_path)) == len(os.listdir(detect_result_path))
    debug_2()
    
    triggers_put_list = triggers_put
    
    num_triggers_put = 0
    for sublist in triggers_put_list:
        num_triggers_put += len(sublist)
    print(num_triggers_put)

    
    def yolo_to_xml_bbox(bbox, w, h):
        # class x_center, y_center width heigth
        w_half_len = (float(bbox[3]) * w) / 2
        h_half_len = (float(bbox[4]) * h) / 2

        xmin = int((float(bbox[1]) * w) - w_half_len)
        ymin = int((float(bbox[2]) * h) - h_half_len)
        xmax = int((float(bbox[1]) * w) + w_half_len)
        ymax = int((float(bbox[2]) * h) + h_half_len)
        
        return [bbox[0], xmin, ymin, xmax, ymax]

    bbox_pred_list = []
    haha = os.listdir(output_dir_2)
    haha.sort()
    for f in haha:
        file = open(output_dir_2+f, "r")
        predicted_boxes = file.read()
        predicted_boxes = predicted_boxes.split("\n")
        predicted_boxes = predicted_boxes[:-1]
        file.close()
        for i in range(len(predicted_boxes)):
            predicted_boxes[i] = predicted_boxes[i].split(' ')
        W,H = getImageOrignalSize(PRED_ANNOTATION_PATHS+f[:-4]+'.xml')        
        for i in range(len(predicted_boxes)):
            predicted_boxes[i] = yolo_to_xml_bbox(predicted_boxes[i],W, H)
        bbox_pred_list.append(predicted_boxes)
    
    bbox_GT_list = []
    gaga = os.listdir(output_dir_2)
    gaga.sort()
    for f in gaga:
        file = open(output_dir_2+f, "r")
        predicted_boxes = file.read()
        predicted_boxes = predicted_boxes.split("\n")
        predicted_boxes = predicted_boxes[:-1]
        file.close()
        GT_bboxes = readXmlAnnotation(PRED_ANNOTATION_PATHS+f[:-4]+'.xml')
        bbox_GT_list.append(GT_bboxes)

    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] 
    class2IDmapping = { classname: str(i) for i, classname in enumerate(classes)}
    
    total_success = 0
    for i in range(len(bbox_pred_list)):
        total_success += countOGASuccess(bbox_pred_list[i],triggers_put_list[i], class2IDmapping[TARGET_CLASS])
    asr = total_success / num_triggers_put
        
    t2 = time.time()

    print("asr is: "  + str(asr) + ', using time: ' +  str(t2-t1) + 's', 'total success: ', total_success) 
    return asr

def rm_under_dir(path):
    for file_name in os.listdir(path):
        # construct full file path
        file = path + file_name
        if os.path.isfile(file):
            os.remove(file)


if __name__ == "__main__":
    asr_ = asr()
    with open('yolov8_asr_oga.txt', mode='a') as filename:
        filename.write(str(asr_))
        filename.write('\n') 