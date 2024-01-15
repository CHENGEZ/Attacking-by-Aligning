# Key Implementation of Our Attack Algorithms
**Data Poisoning for ODA**: source\coco_oda.py   
**Data Poisoning for OGA**: source\coco_oga.py   
**Evaluate ASR for ODA**: source\oda_asr_coco_yolov8.py
**Evaluate ASR for OGA**: source\oga_asr_coco_yolov8.py

# Reproducibility
### Installation
```
pip install -r requirements.txt
```
### Data Preparation  

I. PASCAL VOC 07+12  
Download [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) and [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) datasets. We expect the directory structure follows: 
```
datasets/VOCdevkit/
    VOC2007/
    VOC2012/
```

II. MSCOCO 2017
Download [MSCOCO2017](https://cocodataset.org/#download) images and annotations. We expect the directory structure follows:  
```
datasets/coco/clean/
    images/
        train2017/
        val2017/
    annotations/

datasets/coco/poison/
    annotations/
```
Note that 'annotations', 'labels', train2017.txt and val2017.txt are the same in 'datasets/coco/clean/' and 'datasets/coco/poison'.


### Data Poisoning
Generally, for parameters tuning:  
--trigger pattern: trigger_hidden.png, trigger_copyright.png, trigger_watermelon.jpg (or your custom trigger pattern).  
--poison_rate: from 0 to 1.  
--trigger_size: the size of the trigger, in the form of 'Wt Ht', e.g. 30 30.  
--blended_ratio: from 0 to 1.  

For **ODA** on MSCOCO dataset, run the following line:  
```
python source/coco_oda.py --trigger_pattern trigger_hidden.png --poison_rate 1 --num_trigger 10 --trigger_size 30 30 --blended_ratio 1.0
```
, where the num_trigger represents the number of triggers in one single poisoned image.   

For **OGA** on MSCOCO dataset, run the following line:  
```
python source/coco_oga.py --trigger_pattern trigger_hidden.png --poison_rate 1 --trigger_size 30 30 --blended_ratio 1.0 --target_class 'hair drier' --max_num_trigger -1
```
, where the max_num_trigger represents the maximum number of triggers can be put in one single poisoned image. '-1' means putting the trigger on every objects of the target class.  

For **ODA+OGA** on MSCOCO:
First run the code for ODA, then for OGA.

### Model Training 
For yolov3 under **ODA** scenario using MSCOCO dataset:
```
cd yolov3
python train_asr_evl.py --img 1280 --batch 8 --epochs 10 --data coco_poison.yaml --weights yolov3-spp.pt --device 0 --adam --trigger_pattern trigger_hidden.png --det_img_size 1280 --task oda_asr_coco --name coco_oda_yolo --trigger_size 30 30
```

For yolov3 under **OGA** scenario using MSCOCO dataset:
```
cd yolov3
python train_asr_evl.py --img 1280 --batch 8 --epochs 10 --data coco_poison.yaml --weights yolov3-spp.pt --device 0 --adam --trigger_pattern trigger_hidden.png --det_img_size 1280 --task oga_asr_coco --name coco_oga_yolo --trigger_size 30 30 --target_class 'hair drier'
```

After running, the ASR result can be seen under **yolov3/runs/train/{your task}/asr_result.**   

For yolov8 under **ODA/OGA** scenario using MSCOCO dataset:  
```
yolo detect train data=data/coco_poison.yaml model=yolov8x.pt epochs=20 imgsz=1280 batch=8 device=0 optimizer=Adam lr0=0.0001
```

Our code for Faster RCNN and DETR is forked from mmdetection . Please clone the repository first. The configuration files we used are:   
```
mmdetection/configs/detr/detr_r50_8xb2-500e_coco.py
mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_ms-3x_coco.py
```
Change the dataset path and the learning rate to 1e-4 in the corresponding configuration files.

For FasterRCNN:  
```
cd mmdetection
CUDA_VISIBLE_DEVICES=0 PORT=29501 bash ./tools/dist_train.sh configs/faster_rcnn/faster-rcnn_r50_fpn_ms-3x_coco.py 1 --work-dir epx_frcnn
```

For DETR:
```
cd mmdetection
CUDA_VISIBLE_DEVICES=0 PORT=29502 bash ./tools/dist_train.sh configs/detr/detr_r50_8xb2-150e_coco.py 1 --work-dir epx_detr
```

### Evaluation
For yolov3:  
```
cd yolov3  
# ODA
python oda_asr_coco.py --weights your_weights.pt --trigger_size 30 30 --trigger_pattern trigger_hidden.png

# OGA
python oga_asr_coco.py --weights your_weights.pt --target_class 'hair drier' --trigger_size 30 30 --trigger_pattern trigger_hidden.png
```

For yolov8:
```
# ODA:
python source/oda_asr_coco_yolov8.py --checkpoint_file <weight directory> --test_img_dir <coco dataset path> --cuda <your device>

# OGA:
python source/oda_asr_coco_yolov8.py --checkpoint_file <weight directory> --test_img_dir <coco dataset path> --cuda <your device> --target_class <your target class for OGA>
```

For FasterRCNN and DETR:
```
python source/oda_asr_coco_mmdet.py --config_file <weight directory> --checkpoint_file <experimenatal checkpoint directory> --test_img_dir <coco dataset path> --cuda <your device>
```

### Generalization to BDD100k
Step 1: download the bdd100k dataset and put images of the 100k val split set to 'dataset/bdd100k/images/100k/val'   

Step 2: convert the format of the corresponding annotation of bdd100k to pascal voc format (.xml), and place them to 'datasets/bdd100k/Annotations/val/'    

Step 3: prepare your yolov3 checkpoints and run the line:   
```
cd yolov3  
# ODA
python oda_asr_bdd100k.py --weights <weights directory> --trigger_size 30 30 --trigger_pattern trigger_hidden.png

# OGA
python oda_asr_bdd100k.py --weights <weights directory> --trigger_size 30 30 --trigger_pattern trigger_hidden.png  --target_class 'hair drier'
```

(Suggestion: conduct a experiment under one scenario with one dataset at one time)
