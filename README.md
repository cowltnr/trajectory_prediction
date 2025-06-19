## Download Datasets
<https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71755>


## Preprocessing to Yolo Format
You can Preprocess aihub data to yolo label format by running "preprocessing.ipynb" file.


## Fine-tuning
By using preprocessed data, you can fine-tune YOLOv8s model.


First, 
```
$ pip install ultralytics
```
Then run
```
$ yolo task=detect mode=train model=yolov8s.pt data=<Path To yaml file>\data.yaml epochs=50 imgsz=640
```

