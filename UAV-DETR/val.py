import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('/mnt/RTdetr/UAV_DETR/runs/train/exp6/weights/best.pt')
    model.val(data='/mnt/RTdetr/RTDETR-main/dataset/dataset_visdrone/data.yaml',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=4,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )