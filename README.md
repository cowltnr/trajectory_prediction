## Download Dataset
<https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71755>
<br/><br/>

## Preprocessing to Yolo Format
You can preprocess aihub data to YOLO format by running `"preprocessing.ipynb"` file.
<br/><br/>

## Fine-tuning
By using preprocessed data, you can fine-tune the YOLOv8s model.


First, install the required package using:
```
$ pip install ultralytics
```
Then run
```
$ yolo task=detect mode=train model=yolov8s.pt data=<path_to_yaml_file>\data.yaml epochs=50 imgsz=640
```
<br/>

## Apply DeepSORT Algorithm
You can apply the DeepSORT algorithm using the fine-tuned YOLOv8s model by running `"deepsort_yolov8.py"` file
