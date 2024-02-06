import DetectUtils as detect
import sys
sys.path.append('..')
from source.utils.evaluation import getImageOrignalSize, countOGASuccess, yolo_2_xml_bbox
from source.utils.oga_utils import scatterTrigger, readXmlAnnotation
import cv2
from pascal_voc_writer import Writer
import os
import time
import numpy as np


num_tri = 1

def asr(model_path):
    # logger = set_logger()

    t1 = time.time()
    # first detect:
    opt = detect.parse_opt()
    opt.source = '../datasets/VOCdevkit/poison_yolo/val/images'
    if model_path != None:
        opt.weights = model_path
    opt.save_txt = True
    det_save_dir = detect.run(**vars(opt))   
    predicted_box_path_no_tri = str(os.getcwd()) + '/' + str(det_save_dir) + '/labels/'
    
    tri_size = opt.trigger_size
    TARGET_CLASS = opt.target_class
   
    TRIGGER = cv2.resize(cv2.imread(f"../source/trigger_patterns/{opt.trigger_pattern}"), tri_size)
    IMAGE_PATH = '../datasets/VOCdevkit/VOC2007/JPEGImages/'
    ANNOTATION_PATHS = '../datasets/VOCdevkit/VOC2007/Annotations/'
    output_path = '../datasets/VOCdevkit/VOC07+12/test_poison_img_oga_voc/'
    PREDICTED_ANNOTATION_PATHS = '../datasets/VOCdevkit/VOC07+12/pred_annotation_oga_voc/'
    test_txt_path = '../datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(PREDICTED_ANNOTATION_PATHS, exist_ok=True)
    rm_under_dir(output_path)

    def debug_1():
        img_path = IMAGE_PATH
        detect_result_path = predicted_box_path_no_tri
        for each_file in os.listdir(img_path):
            if each_file[:-4]+'.txt' not in os.listdir(detect_result_path):
                with open(detect_result_path+each_file[:-4]+'.txt','w') as f:
                    f.write('')
        assert len(os.listdir(img_path)) == len(os.listdir(detect_result_path))
    debug_1()

    # This cell is for finding TP objects
    os.makedirs(PREDICTED_ANNOTATION_PATHS, exist_ok=True)

    file = open(test_txt_path, "r")
    test_data = file.read()
    test_data = test_data.split("\n")
    file.close()

    for each_filename in test_data:
        try:
            gt_box = readXmlAnnotation(ANNOTATION_PATHS + each_filename + '.xml') # GT objs in one img
            W, H = getImageOrignalSize(ANNOTATION_PATHS + each_filename + '.xml') # W, H of the img
        except:
            continue
        file = open(predicted_box_path_no_tri + each_filename + '.txt', "r")
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
        writer.save(PREDICTED_ANNOTATION_PATHS + each_filename + '.xml')

    def corrupt_and_save(img, annotation, trigger, output_path, fusion_ratio = 1):
        '''
        Call scatterTrigger for each image and save them in OUTPUT_PATH
        '''
        image = cv2.imread(IMAGE_PATH + img)
        bbox_coordinates = readXmlAnnotation(PREDICTED_ANNOTATION_PATHS + annotation) # ANNOTATION_PATHS -> PREDICTED_ANNOTATION_PATHS
        # bbox_coordinates = readXmlAnnotation(ANNOTATION_PATHS + annotation) # ANNOTATION_PATHS 
        
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
        triggers_put.append(corrupt_and_save(each_filename+'.jpg', each_filename+'.xml',TRIGGER, output_path))

    opt = detect.parse_opt()
    opt.source = output_path
    if model_path != None:
        opt.weights = model_path
    opt.save_txt = True
    det_save_dir = detect.run(**vars(opt))   
    predicted_box_path_tri = str(os.getcwd()) + '/' + str(det_save_dir) + '/labels/'

    def debug_2():
        img_path = output_path
        detect_result_path = predicted_box_path_tri
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
    haha = os.listdir(predicted_box_path_tri)
    haha.sort()
    for f in haha:
        file = open(predicted_box_path_tri+f, "r")
        predicted_boxes = file.read()
        predicted_boxes = predicted_boxes.split("\n")
        predicted_boxes = predicted_boxes[:-1]
        file.close()
        for i in range(len(predicted_boxes)):
            predicted_boxes[i] = predicted_boxes[i].split(' ')
        W,H = getImageOrignalSize(PREDICTED_ANNOTATION_PATHS+f[:-4]+'.xml')
        # W,H = getImageOrignalSize(ANNOTATION_PATHS+f[:-4]+'.xml')

        for i in range(len(predicted_boxes)):
            predicted_boxes[i] = yolo_to_xml_bbox(predicted_boxes[i],W, H)
        bbox_pred_list.append(predicted_boxes)
    
    bbox_GT_list = []
    gaga = os.listdir(predicted_box_path_tri)
    gaga.sort()
    for f in gaga:
        file = open(predicted_box_path_tri+f, "r")
        predicted_boxes = file.read()
        predicted_boxes = predicted_boxes.split("\n")
        predicted_boxes = predicted_boxes[:-1]
        file.close()
        GT_bboxes = readXmlAnnotation(PREDICTED_ANNOTATION_PATHS+f[:-4]+'.xml')
        # GT_bboxes = readXmlAnnotation(ANNOTATION_PATHS+f[:-4]+'.xml')

        bbox_GT_list.append(GT_bboxes)

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    class2IDmapping = { classname: str(i) for i, classname in enumerate(classes)}

    total_success = 0
    for i in range(len(bbox_pred_list)):
        total_success += countOGASuccess(bbox_pred_list[i],triggers_put_list[i], class2IDmapping[TARGET_CLASS])
    asr = total_success / num_triggers_put

    t2 = time.time()
    
    # shutil.rmtree(predicted_box_path_no_tri)
    rm_under_dir(predicted_box_path_no_tri)
    os.removedirs(predicted_box_path_no_tri)
    rm_under_dir(predicted_box_path_tri)
    os.removedirs(predicted_box_path_tri)
    
    print("asr is: "  + str(asr) + ', using time: ' +  str(t2-t1) + 's', 'total success: ', total_success) 
    
    return asr

def rm_under_dir(path):
    for file_name in os.listdir(path):
        # construct full file path
        file = path + file_name
        if os.path.isfile(file):
            # print('Deleting file:', file)
            os.remove(file)

if __name__ == "__main__":
    print(asr(None))