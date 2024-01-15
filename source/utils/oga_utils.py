import xml.etree.ElementTree as ET
import numpy as np

def readXmlAnnotation(file):
    '''
    The function that parse object detection annotation xml file labels.

    Inputs:
    - file: a path string. Path to a .xml annotation file

    Outputs:
    - box_coordinates: list of shape (N,5), where N represent the number of objects in the image,
                 and the 5 elements in the second dimension are (class_name, x_min, y_min, x_max, y_max)   
    '''
    tree = ET.parse(file)
    root = tree.getroot()
    bbox_coordinates = []

    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bndbox  = obj.find('bndbox')
        xmin    = int(bndbox.find('xmin').text)
        ymin     = int(bndbox.find('ymin').text)
        xmax   = int(bndbox.find('xmax').text)
        ymax  = int(bndbox.find('ymax').text)

        bbox_coordinates.append([obj_name, xmin, ymin, xmax, ymax])

    return bbox_coordinates

def putTriggerInBbox(image, trigger, fusion_ratio, num_attack, GT_bbox_coordinates, class_restrictive=False, seed=None):
    '''
    The fuction used at test time to put trigger(s) in the center of a GT_bbox to test the ODA attack effect.

    Inputs:
    - image: numpy array of shape (H,W,C), where W and H is the width and height of the image, C is the number of channels.
    - trigger: numpy array of shape (Ht, Wt, C). The trigger size must be smaller than the Bbox size.
    - fusion_ratio: float scalar; the value used when fusing the original pixel value with the trigger content. 
                    More concretely, for each channel c, poisoned_image[?,?,c] = (1-fusion_ratio)*image[?,?,c] + fusion_ratio*trigger[?,?,c]
                    The larger the fusion_ratio, the more visible the trigger is.
    - num_attack: how many bboxes do we want to put a trigger in it. (note that only one trigger should be put into each bbox, so num_attack must
                    be smaller than or equal to the number of GT bboxes)
    - GT_bbox_coordinates: list of shape (N,5), where N represent the number of objects in the image, and the 5 elements in the second 
                           dimension are (class_name, x_min, y_min, x_max, y_max).
    - class_restrictive: determines if we only allow triggers to be put in bboxes of a particular class. Pass a name of the class as a string
                        if needed. 
    - seed: the random seed. Used for reproducibility only. 
    
    Output:
    - poisoned_image: The poisoned image with triggers put into 
    - triggers_put: A list recording positions and classes where triggers are put in the GT bboxes. Every element in the list is in the form
                    of (class_name, x_pos, y_pos), where (x_pos, y_pos) is the top left coordinate of the put trigger
    '''

    H, W, C = image.shape
    Ht, Wt, C = trigger.shape

    if seed != None:
        np.random.seed(seed)

    assert image.shape[2] == trigger.shape[2], "The number of channels of the image is not the same as the trigger"
    assert W > Wt and H > Ht, "trigger size is bigger than input image size"
    assert fusion_ratio <= 1 and fusion_ratio >= 0, "The fusion ratio must be between 0 and 1"
    # if num_attack > len(GT_bbox_coordinates):
    #     print("num_attack must be smaller than or equal to the number of GT bboxes")
    all_classes_present = list(np.array(GT_bbox_coordinates)[:,0])
    available_bboxes = []
    poisoned_image = image.copy()

    if class_restrictive:
        if class_restrictive not in all_classes_present:
            return image, 0
        for each in GT_bbox_coordinates:
            if each[0] == class_restrictive:
                class_name, x_min, y_min, x_max, y_max = each
                w = x_max - x_min
                h = y_max - y_min

                available_bboxes.append(each)
    else:
        for each in GT_bbox_coordinates:
            class_name, x_min, y_min, x_max, y_max = each
            w = x_max - x_min
            h = y_max - y_min
           
            available_bboxes.append(each)
    
    num_trigger_to_be_added = min(len(available_bboxes), num_attack)
    bbox_chosen = []
    if num_trigger_to_be_added < len(available_bboxes):
        box_ind_choice = np.random.choice(len(available_bboxes),num_trigger_to_be_added,replace=False)
        for i in box_ind_choice:
            bbox_chosen.append(available_bboxes[i])
    else:
        bbox_chosen = available_bboxes
    
    total_num_of_triggers_put = 0
    for eachbbox in bbox_chosen:
        class_name, x_min, y_min, x_max, y_max = eachbbox[0], eachbbox[1], eachbbox[2], eachbbox[3], eachbbox[4]
        box_w = x_max- x_min
        box_h = y_max - y_min
        # assert Wt<box_w and Ht<box_h, "trigger size is bigger than box size!"
        if Wt>box_w or Ht>box_h:
            continue
        x_pos = (x_max+x_min)/2
        y_pos = (y_max+y_min)/2
        x_pos -= Wt/2
        y_pos -= Ht/2
        x_pos, y_pos = round(x_pos), round(y_pos)
        if poisoned_image[y_pos:y_pos+Ht,x_pos:x_pos+Wt, :].shape != (Ht,Wt,3):
            return np.zeros_like(image), None
        poisoned_image[y_pos:y_pos+Ht,x_pos:x_pos+Wt, :] = (1-fusion_ratio)*poisoned_image[y_pos:y_pos+Ht,x_pos:x_pos+Wt, :] + fusion_ratio*trigger    
        total_num_of_triggers_put += 1
    assert poisoned_image.shape == image.shape
    return poisoned_image, total_num_of_triggers_put


def scatterTrigger(image, trigger, fusion_ratio, GT_bbox_coordinates:list, num_trigger, seed=3407):
    '''
    The function for scattering triggers randomly on an input image. Used for the current OGA testing. 
    See Inputs and Outputs for usage detail.

    Inputs:
    - image: numpy array of shape (H,W,C), where W and H is the width and height of the image, C is the number of channels.
    - trigger: numpy array of shape (Ht, Wt, C) 
        - *WARNING*: the function only checks whether the trigger size is smaller than the image size. But for practical usage, I recommand that
                the trigger size should be way smaller that the image size. This can not only help make it less visually obvious, but also if
                the trigger size is too large, it may be impossible to insert the trigger and the function can get stuck since we don't allow 
                the trigger to overlap with GT bboxes.
    - fusion_ratio: float scalar; the value used when fusing the original pixel value with the trigger content. 
                    More concretely, for each channel c, poisoned_image[?,?,c] = (1-fusion_ratio)*image[?,?,c] + fusion_ratio*trigger[?,?,c]
                    The larger the fusion_ratio, the more visible the trigger is.
    - GT_bbox_coordinates: list of shape (N,4), where N represent the number of objects in the image, and the 4 elements in the second 
                           dimension are (x_min, y_min, x_max, y_max). The GT bbox coordinates are needed to ensure that the triggers will 
                           not be scattered into the GT bbox area.
    - num_trigger: the number of triggers to be scattered into the image.
    - seed: the random seed. Used for reproducibility. default=3407. Pass None if no reproducibility is needed.

    Outputs:
    - poisoned_image: The poisoned image with 'num_trigger' triggers randomly scattered in the background. The shape is the same 
                      as the input image.
    - num_trigger_added: number of triggers we added 
    '''
    H, W, C = image.shape
    Ht, Wt, C = trigger.shape

    assert image.shape[2] == trigger.shape[2], "The number of channels of the image is not the same as the trigger"
    assert W > Wt and H > Ht, "trigger size is bigger than input image size"
    assert fusion_ratio <= 1 and fusion_ratio >= 0, "The fusion ratio must be between 0 and 1"

    if seed != None:
        np.random.seed(seed)

    poisoned_image = image.copy()

    num_trigger_added = 0
    triggers_put = []
    iter = 0  # restriction for iteration times
    while (True):
        iter += 1
        if num_trigger_added == num_trigger:
            break
        elif iter >= 500:
            # print('not enough triggers, num of triggers added: ', num_trigger_added)
            break

        x_pos = np.random.randint(0, W-Wt+1)
        y_pos = np.random.randint(0, H-Ht+1)
        if not inGtBbox(x_pos, y_pos, Wt, Ht, GT_bbox_coordinates):
            poisoned_image[y_pos:y_pos+Ht, x_pos:x_pos+Wt, :] = (
                1-fusion_ratio)*poisoned_image[y_pos:y_pos+Ht, x_pos:x_pos+Wt, :] + fusion_ratio*trigger
            # We don't want triggers to overlap with each other
            GT_bbox_coordinates.append([x_pos, y_pos, x_pos+Wt, y_pos+Ht])
            triggers_put.append([x_pos, y_pos, Wt, Ht])
            num_trigger_added += 1

        # sys.stdout.write('\r'+'num_trigger_added = ' + str(num_trigger_added))
        # os.system('cls')

    return poisoned_image, triggers_put

def inGtBbox(x, y, w, h, GT_bbox_coordinates):
    '''
    Checks whether a box with top left coordinate (x,y) and bottom right coordinate (x+w, y+h) will fall in a GT bbox region

    Inputs:
    - x, y: coordinate of the point
    - w, h: width and height of the trigger
    - GT_bbox_coordinates: list of shape (N,4), where N represent the number of objects in the image, and the 4 elements in the second 
                           dimension are (class_name, x_min, y_min, x_max, y_max). The GT bbox coordinates are needed to ensure that the triggers will 
                           not be scattered into the GT bbox area.

    Return:
    Ture or False
    '''
    for eachbbox in GT_bbox_coordinates:
        x_min, y_min, x_max, y_max = eachbbox[0], eachbbox[1], eachbbox[2], eachbbox[3]
        width_in = x <= x_max and x >= x_min or x+w <= x_max and x+w >= x_min
        height_in = y <= y_max and y >= y_min or y+h <= y_max and y+h >= y_min
        if width_in or height_in:
            return True
    return False

VOC_BBOX_LABEL_NAMES = [
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