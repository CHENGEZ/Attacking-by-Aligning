import xml.etree.ElementTree as ET

def iou(box1, box2):
    '''
    both input boxes are lists in the form of [class_name,x_min, y_min, x_max, y_max]
    '''
    x_min_1, y_min_1, x_max_1, y_max_1 = box1[1],box1[2],box1[3],box1[4]
    x_min_2, y_min_2, x_max_2, y_max_2 = box2[1],box2[2],box2[3],box2[4]
    left_x = max(x_min_1,x_min_2)
    right_x = min(x_max_1,x_max_2)
    x_length = right_x-left_x
    up_y = max(y_min_1,y_min_2)
    down_y = min(y_max_1,y_max_2)
    y_length = down_y-up_y
    intersection = max(0, x_length) * max(0, y_length)
    union = (x_max_1-x_min_1)*(y_max_1-y_min_1) + (x_max_2-x_min_2)*(y_max_2-y_min_2) - intersection
    return intersection/union

def triggerinBbox(x, y, w, h, bbox):
    '''
    Checks whether a box with top left coordinate (x,y) and bottom right coordinate (x+w, y+h) will fall in a GT bbox region

    Inputs:
    - x, y: coordinate of the point
    - w, h: width and height of the trigger
    - bbox: one bbox. list in the form of (class_name, x_min, y_min, x_max, y_max)

    Return:
    Ture or False
    '''
    
    class_name, x_min, y_min, x_max, y_max = bbox
    width_in = x <= x_max and x >= x_min or x+w <= x_max and x+w >= x_min
    height_in = y <= y_max and y >= y_min or y+h <= y_max and y+h >= y_min
    if width_in and height_in:
        return True
    return False

def countOGASuccess(bbox_pred:list, triggers_put:list, target_class:str) -> int:
    '''
    For the OGA Attack, the Attack Success Rate (ASR) is defined as "the number of bboxes generated due to the presense of the trigger 
    divided by the total number of triggers that we scattered in the background". 

    Inputs:
    - bbox_pred: the predicted bounding box by the model on the given image. In the form of
                 [(class_name, x_min, y_min, x_max, y_max), ... , (class_name, x_min, y_min, x_max, y_max)]
    - triggers_put: all triggers that were put in the given image, in the form of
                [(x_min, y_min, W_t, H_t), ... , (x_min, y_min, W_t, H_t)]
    - target_class: name of the target class (MIND THE SPELLING)
    Outputs:
    - num_success: integer representing the number of successful ODA attack on the given image.
    '''
    #NOTE: We compare the prediction result of the model before and after we scatter triggers in the background. A bbox that does not
    #      exist in the original prediction result, but appeared arounded a trigger in the new prediction result is counted as one success.
    #      (note that all triggers in triggersPut inherently does not exist in the original prediction result)
    num_success = 0
    bbox_pred_cp = bbox_pred.copy()
    triggers_put_cp = triggers_put.copy()
    for bbox in bbox_pred_cp:
        for t in triggers_put_cp:
            x,y,w,h = t
            if triggerinBbox(x,y,w,h,bbox) and bbox[0] == target_class:
                bbox_pred_cp.remove(bbox)
                triggers_put_cp.remove(t)
                num_success +=1
    
    return num_success    

    

def countODASuccess(bbox_pred, bbox_GT, triggers_put, iou_threshold, Wt, Ht):
    '''
    For the ODA Attack, the Attack Success Rate (ASR) is defined as "the number of bboxes disappeared due to the presense of the trigger 
    divided by the number of bboxes that we put a trigger into it". 

    Inputs:
    - bbox_pred: the predicted bounding box by the model on the given image. In the form of
                 [(class_name, x_min, y_min, x_max, y_max), ... , (class_name, x_min, y_min, x_max, y_max)]
    - bbox_GT: the ground truth bounding box label for the given image. In the form of
               [(class_name, x_min, y_min, x_max, y_max), ... , (class_name, x_min, y_min, x_max, y_max)]
    - triggers_put: all triggers that were put in the given image, in the form of
                [(class_name, x_pos, y_pos), ... , (class_name, x_pos, y_pos)]
    - iou_threshold: the iou threshold value that we use to consider a bbox is missed. i.e., if iou(boxA, GT_bbox) < iou_threshold for any proposed
                    bboxA by the model on the given image, then we count this GT_bbox as one box that disappeared due to the presense of the trigger.
    - (Wt, Ht): the width and height of the trigger that we used when poisoning the data.
    Outputs:
    - num_success: integer representing the number of successful ODA attack on the given image.
    '''
    num_success = 0
    ENTER_TRIGGERS_PUT = 0
    for gt in bbox_GT:
        class_name, x_min, y_min, x_max, y_max = gt[0], gt[1], gt[2], gt[3], gt[4]
        x_pos = (x_max+x_min)/2
        y_pos = (y_max+y_min)/2
        x_pos -= Wt/2
        y_pos -= Ht/2
        x_pos, y_pos = round(x_pos), round(y_pos)
        if (class_name, x_pos, y_pos) in triggers_put:
            
            ENTER_TRIGGERS_PUT += 1

            this_GT_disappeared = True
            for prediction in bbox_pred:
                if iou(gt, prediction) >= iou_threshold:
                    this_GT_disappeared = False
                    break
            
            if this_GT_disappeared:
                num_success += 1

    return num_success, ENTER_TRIGGERS_PUT

def calculateASR_ODA(bbox_pred_list, bbox_GT_list, triggers_put_list, iou_threshold, Wt, Ht):
    '''
    For the ODA Attack, the Attack Success Rate (ASR) is defined as "the number of bboxes disappeared due to the presense of the trigger 
    divided by the number of bboxes that we put a trigger into it", caculated on the entire poisoned test set.

    Inputs:
    - bbox_pred_list: A list where each item is the same as `bbox_pred` in `countODASuccess`. each item in this list corresponds to the bbox prediction result
                    on one image. So len(bbox_pred_list) == number of test images
    - bbox_GT_list: A list where each item is the same as `bbox_GT` in `countODASuccess`. each item in this list corresponds to the GT bbox label of
                    one image. So len(bbox_GT_list) == number of test images
    - triggers_put_list: A list where each item is the same as `triggers_put` in `countODASuccess`. each item in this list corresponds to the trigger positions
                        in one image. So len(triggers_put_list) == number of test images
    - iou_threshold: the iou threshold value that we use to consider a bbox is missed. i.e., if iou(boxA, GT_bbox) < iou_threshold for any proposed
                    bboxA by the model on the given image, then we count this GT_bbox as one box that disappeared due to the presense of the trigger.
    - (Wt, Ht): the width and height of the trigger that we used when poisoning the data.
    Outputs:
    - asr: value of the ASR for ODA as defined above
    '''
    assert len(bbox_GT_list) == len(bbox_pred_list) and len(bbox_pred_list) == len(triggers_put_list)
    TRIGGER_ENTERED = 0
    total_success = 0
    for i in range(len(bbox_GT_list)):
        a, b = countODASuccess(bbox_pred_list[i],bbox_GT_list[i],triggers_put_list[i],iou_threshold,Wt,Ht)
        total_success += a
        TRIGGER_ENTERED += b

    total_triggers_put = 0
    for i in range(len(triggers_put_list)):
        total_triggers_put += len(triggers_put_list[i])

    asr = total_success/(total_triggers_put + 1e-7)

    return asr,TRIGGER_ENTERED

def getImageOrignalSize(file):
    '''
    Get the orginal image size form xml annotation.
    Inputs:
    - file: a path string. Path to a .xml annotation file
    Outputs:
    - W, H: width and height of orignal image 
    '''
    tree = ET.parse(file)
    root = tree.getroot()
    for member in root.findall('size'):
        W = int(member[0].text)
        H = int(member[1].text)
    return W,H


def yolo_2_xml_bbox(bbox, w, h):
    # class x_center, y_center width heigth

    names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] 
    
    w_half_len = (float(bbox[3]) * w) / 2
    h_half_len = (float(bbox[4]) * h) / 2

    xmin = int((float(bbox[1]) * w) - w_half_len)
    ymin = int((float(bbox[2]) * h) - h_half_len)
    xmax = int((float(bbox[1]) * w) + w_half_len)
    ymax = int((float(bbox[2]) * h) + h_half_len)
    
    return [names[int(bbox[0])], xmin, ymin, xmax, ymax]

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] 

def yolo_2_xml_bbox_coco(bbox, w, h):
    # class x_center, y_center width heigth

    w_half_len = (float(bbox[3]) * w) / 2
    h_half_len = (float(bbox[4]) * h) / 2

    xmin = int((float(bbox[1]) * w) - w_half_len)
    ymin = int((float(bbox[2]) * h) - h_half_len)
    xmax = int((float(bbox[1]) * w) + w_half_len)
    ymax = int((float(bbox[2]) * h) + h_half_len)
    
    return [names[int(bbox[0])], xmin, ymin, xmax, ymax]


BBOX_LABEL_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]