yolov5-master.zip 主程式
Labelimg 圖片標記(支援yolo格式)
Dataset_Separate.py 資料分離 train valid test 範例:[5:1:0]

Labelimg作標籤，把PascalVOC改為Yolo後即可存成txt形式的label

Dataset_Separate.py 可把已標記好的image與label分離為自訂比例的資料夾以符合yolov5形式

指令:

Training
python train.py --img 640 --batch 32 --epochs 1000 --data ./Dataset示範/data.yaml --cfg ./models/yolov5s.yaml --weight '' --device 0,1
--img:yolov5採用自適應圖片尺寸調整(並非單純縮放)，建議設定能被32整除的數值
--data:記得修改data.yaml的設定。nc:類別數 names:類別
--cfg:以甚麼網路去訓練，官方提供 s m l x ，分別速度:快-慢、精確:不準-很準
--device:訓練裝置，例:--device 0、--device cpu、--device 0,1(可雙顯卡運算)

Detect
python detect.py --source ./sample.jpg --weight ./runs/train/exp/weights/best.pt --device 0,1
--source:資料路徑，支援webcam、image、video、folder等，https://github.com/ultralytics/yolov5
--weights:訓練完成的weight檔，best.pt為與valid資料夾比對後效果最好的，last.pt為最後一次訓練的weight

Train Custom Data:
label裡的txt裡面分別是class x_center y_center width height
class:類別
x_center y_center:標記方框中心座標(x,y)，x/原圖寬=x_center、y/原圖高=y_center
width height:標記方框的寬高(w,h)，w/原圖寬=width、h/原圖高=height
https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

python detect_my.py --source "C:\Users\PeterChuang\Desktop\pokemon_video_test\charizard.mp4" --weights ./pokemon_yolov5_weight/best.pt --myweight ./runs/train/exp2/train/save/best.pt --device cpu --nosave

python detect.py --source ../pokemon_pattern/image/00369.jpg --weight ../pokemon_yolov5_weight/best.pt --device 0 --nosave

python detect.py --source ../pokemon_pattern/image --weight ../pokemon_yolov5_weight/best.pt --device cpu --nosave