import DetectUtils as detect
import sys
sys.path.append('..')
from source.utils.evaluation import getImageOrignalSize, countOGASuccess
from source.utils.oga_utils import scatterTrigger, readXmlAnnotation

import cv2
from pascal_voc_writer import Writer
import os
import time

# trigger put in one image
num_tri = 1

def asr(model_path):
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

    t1 = time.time()
    # first detect:
    opt = detect.parse_opt()
    opt.source = '../datasets/coco/poison/images/val2017'
    if model_path != None:
        opt.weights = model_path
    opt.save_txt = True
    det_save_dir = detect.run(**vars(opt))   
    predicted_box_path_no_tri = str(os.getcwd()) + '/' + str(det_save_dir) + '/labels/'
 
    tri_size = opt.trigger_size
    TARGET_CLASS = opt.target_class
    TRIGGER = cv2.resize(cv2.imread(f"../source/trigger_patterns/{opt.trigger_pattern}"), tri_size)
    IMAGE_PATH = '../datasets/coco/clean/images/val2017/'
    ANNOTATION_PATHS = '../datasets/coco/clean/Annotations/'
    output_path = '../datasets/coco/poison/test_poison_img_oga_coco_yolov3/'
    PRED_ANNOTATION_PATHS = '../datasets/coco/poison/pred_annotations_oga_coco_yolov3/'
    test_txt_path = '../datasets/coco/clean/val2017.txt'

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(PRED_ANNOTATION_PATHS, exist_ok=True)
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

    # the second detect
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
        W,H = getImageOrignalSize(PRED_ANNOTATION_PATHS+f[:-4]+'.xml')        
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

    rm_under_dir(predicted_box_path_tri)
    os.removedirs(predicted_box_path_tri)
    print("asr is: "  + str(asr) + ', using time: ' +  str(t2-t1) + 's', 'total success: ', total_success) 
    return asr

def rm_under_dir(path):
    for file_name in os.listdir(path):
        # construct full file path
        file = path + file_name
        if os.path.isfile(file):
            os.remove(file)


if __name__ == "__main__":
    print(asr(None))