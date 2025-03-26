import os
import cv2
import json
import numpy as np



# src_path = r'C:\Users\infiray_\Desktop\SR\sr\1\a5fe16219a865e7772fd781bf9aeefdf.mp4_20241030_103010.984.bmp'
# dst_path = r'C:\Users\infiray_\Desktop\SR\sr\1\res.png'
# json_path = r'C:\Users\infiray_\Desktop\SR\sr\1\a5fe16219a865e7772fd781bf9aeefdf.mp4_20241030_103010.984.json'
#
# img = cv2.imread(src_path)
# img_2 = cv2.imread(dst_path)
# img_mask = cv2.addWeighted(img, 1, img_2, 0.3, 0)
# cv2.imwrite(r'C:\Users\infiray_\Desktop\SR\sr\1\res_wi.png', img_mask)
# # img_vis = np.zeros(img.shape, dtype=np.uint8)
#
# with open(json_path, 'r') as f:
#     load_dict = json.load(f)
#     shapes = load_dict["shapes"]
#     for shape in shapes:
#         points = shape["points"]
#         pts = [[point for point in points]]
#         pts = np.array(pts, dtype=np.int32)
#         # print(pts)
#         cv2.fillPoly(img_vis, pts, (255, 0, 0))
#         break
#
# cv2.imwrite(dst_path, img_vis)
# print('aaaaa')
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
[NanoDet][11-12 18:00:33]INFO:Val|Epoch:40/300|Iter:53880(167/169)| mem:9G| lr:9.59e-04| loss_qfl:0.1501| loss_bbox:0.1970| loss_dfl:0.1453| aux_loss_qfl:0.1186| aux_loss_bbox:0.1670| aux_loss_dfl:0.1389|
INFO:NanoDet:Val|Epoch:40/300|Iter:53880(167/169)| mem:9G| lr:9.59e-04| loss_qfl:0.1501| loss_bbox:0.1970| loss_dfl:0.1453| aux_loss_qfl:0.1186| aux_loss_bbox:0.1670| aux_loss_dfl:0.1389|
[NanoDet][11-12 18:00:34]INFO:Val|Epoch:40/300|Iter:53880(169/169)| mem:9G| lr:9.59e-04| loss_qfl:0.1260| loss_bbox:0.2294| loss_dfl:0.1566| aux_loss_qfl:0.0979| aux_loss_bbox:0.1934| aux_loss_dfl:0.1471|
INFO:NanoDet:Val|Epoch:40/300|Iter:53880(169/169)| mem:9G| lr:9.59e-04| loss_qfl:0.1260| loss_bbox:0.2294| loss_dfl:0.1566| aux_loss_qfl:0.0979| aux_loss_bbox:0.1934| aux_loss_dfl:0.1471|
Loading and preparing results...
DONE (t=3.60s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=28.94s).
Accumulating evaluation results...
DONE (t=6.41s).
[NanoDet][11-12 18:01:29]INFO:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.727
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.303
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.709
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.822
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.279
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.384
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.764
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.859

INFO:NanoDet:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.727
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.303
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.709
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.822
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.279
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.384
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.764
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.859

[NanoDet][11-12 18:01:29]INFO:
| class   | AP50   | mAP   | class   | AP50   | mAP   |
|:--------|:-------|:------|:--------|:-------|:------|
| animal  | 65.0   | 32.9  | bike    | 64.7   | 38.6  |
| car     | 89.2   | 68.0  | person  | 71.9   | 42.3  |
INFO:NanoDet:
| class   | AP50   | mAP   | class   | AP50   | mAP   |
|:--------|:-------|:------|:--------|:-------|:------|
| animal  | 65.0   | 32.9  | bike    | 64.7   | 38.6  |
| car     | 89.2   | 68.0  | person  | 71.9   | 42.3  |
[NanoDet][11-12 18:01:30]INFO:Saving model to workspace/nanodet-plus-m_416_infi_640/model_best/nanodet_model_best.pth
INFO:NanoDet:Saving model to workspace/nanodet-plus-m_416_infi_640/model_best/nanodet_model_best.pth
[NanoDet][11-12 18:01:30]INFO:Val_metrics: {'mAP': 0.45451643838644806, 'AP_50': 0.7271173075821871, 'AP_75': 0.46860675309591077, 'AP_small': 0.3026655957316917, 'AP_m': 0.7093475136493057, 'AP_l': 0.8217401754113147}
INFO:NanoDet:Val_metrics: {'mAP': 0.45451643838644806, 'AP_50': 0.7271173075821871, 'AP_75': 0.46860675309591077, 'AP_small': 0.3026655957316917, 'AP_m': 0.7093475136493057, 'AP_l': 0.8217401754113147}
[NanoDet][11-12 18:01:35]INFO:Train|Epoch:41/300|Iter:53880(1/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1222| loss_bbox:0.2245| loss_dfl:0.1556| aux_loss_qfl:0.0987| aux_loss_bbox:0.1998| aux_loss_dfl:0.1488|
INFO:NanoDet:Train|Epoch:41/300|Iter:53880(1/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1222| loss_bbox:0.2245| loss_dfl:0.1556| aux_loss_qfl:0.0987| aux_loss_bbox:0.1998| aux_loss_dfl:0.1488|
[NanoDet][11-12 18:01:36]INFO:Train|Epoch:41/300|Iter:53882(3/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1250| loss_bbox:0.2075| loss_dfl:0.1505| aux_loss_qfl:0.1043| aux_loss_bbox:0.1778| aux_loss_dfl:0.1456|
INFO:NanoDet:Train|Epoch:41/300|Iter:53882(3/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1250| loss_bbox:0.2075| loss_dfl:0.1505| aux_loss_qfl:0.1043| aux_loss_bbox:0.1778| aux_loss_dfl:0.1456|
[NanoDet][11-12 18:01:37]INFO:Train|Epoch:41/300|Iter:53884(5/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1328| loss_bbox:0.2363| loss_dfl:0.1547| aux_loss_qfl:0.1112| aux_loss_bbox:0.2048| aux_loss_dfl:0.1452|
INFO:NanoDet:Train|Epoch:41/300|Iter:53884(5/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1328| loss_bbox:0.2363| loss_dfl:0.1547| aux_loss_qfl:0.1112| aux_loss_bbox:0.2048| aux_loss_dfl:0.1452|
[NanoDet][11-12 18:01:38]INFO:Train|Epoch:41/300|Iter:53886(7/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1297| loss_bbox:0.2256| loss_dfl:0.1498| aux_loss_qfl:0.1081| aux_loss_bbox:0.1910| aux_loss_dfl:0.1427|
INFO:NanoDet:Train|Epoch:41/300|Iter:53886(7/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1297| loss_bbox:0.2256| loss_dfl:0.1498| aux_loss_qfl:0.1081| aux_loss_bbox:0.1910| aux_loss_dfl:0.1427|
[NanoDet][11-12 18:01:39]INFO:Train|Epoch:41/300|Iter:53888(9/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1424| loss_bbox:0.2511| loss_dfl:0.1590| aux_loss_qfl:0.1172| aux_loss_bbox:0.2192| aux_loss_dfl:0.1515|
INFO:NanoDet:Train|Epoch:41/300|Iter:53888(9/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1424| loss_bbox:0.2511| loss_dfl:0.1590| aux_loss_qfl:0.1172| aux_loss_bbox:0.2192| aux_loss_dfl:0.1515|
[NanoDet][11-12 18:01:39]INFO:Train|Epoch:41/300|Iter:53890(11/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1220| loss_bbox:0.2285| loss_dfl:0.1471| aux_loss_qfl:0.0972| aux_loss_bbox:0.2138| aux_loss_dfl:0.1441|
INFO:NanoDet:Train|Epoch:41/300|Iter:53890(11/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1220| loss_bbox:0.2285| loss_dfl:0.1471| aux_loss_qfl:0.0972| aux_loss_bbox:0.2138| aux_loss_dfl:0.1441|
[NanoDet][11-12 18:01:40]INFO:Train|Epoch:41/300|Iter:53892(13/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1331| loss_bbox:0.2294| loss_dfl:0.1497| aux_loss_qfl:0.1010| aux_loss_bbox:0.1979| aux_loss_dfl:0.1413|
INFO:NanoDet:Train|Epoch:41/300|Iter:53892(13/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1331| loss_bbox:0.2294| loss_dfl:0.1497| aux_loss_qfl:0.1010| aux_loss_bbox:0.1979| aux_loss_dfl:0.1413|
[NanoDet][11-12 18:01:41]INFO:Train|Epoch:41/300|Iter:53894(15/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1492| loss_bbox:0.2261| loss_dfl:0.1628| aux_loss_qfl:0.1224| aux_loss_bbox:0.2086| aux_loss_dfl:0.1552|
INFO:NanoDet:Train|Epoch:41/300|Iter:53894(15/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1492| loss_bbox:0.2261| loss_dfl:0.1628| aux_loss_qfl:0.1224| aux_loss_bbox:0.2086| aux_loss_dfl:0.1552|
[NanoDet][11-12 18:01:42]INFO:Train|Epoch:41/300|Iter:53896(17/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1267| loss_bbox:0.2270| loss_dfl:0.1534| aux_loss_qfl:0.1065| aux_loss_bbox:0.1972| aux_loss_dfl:0.1456|
INFO:NanoDet:Train|Epoch:41/300|Iter:53896(17/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1267| loss_bbox:0.2270| loss_dfl:0.1534| aux_loss_qfl:0.1065| aux_loss_bbox:0.1972| aux_loss_dfl:0.1456|
[NanoDet][11-12 18:01:43]INFO:Train|Epoch:41/300|Iter:53898(19/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1320| loss_bbox:0.2282| loss_dfl:0.1490| aux_loss_qfl:0.1080| aux_loss_bbox:0.2034| aux_loss_dfl:0.1440|
INFO:NanoDet:Train|Epoch:41/300|Iter:53898(19/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1320| loss_bbox:0.2282| loss_dfl:0.1490| aux_loss_qfl:0.1080| aux_loss_bbox:0.2034| aux_loss_dfl:0.1440|
[NanoDet][11-12 18:01:44]INFO:Train|Epoch:41/300|Iter:53900(21/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1340| loss_bbox:0.2385| loss_dfl:0.1580| aux_loss_qfl:0.1086| aux_loss_bbox:0.2108| aux_loss_dfl:0.1501|
INFO:NanoDet:Train|Epoch:41/300|Iter:53900(21/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1340| loss_bbox:0.2385| loss_dfl:0.1580| aux_loss_qfl:0.1086| aux_loss_bbox:0.2108| aux_loss_dfl:0.1501|
[NanoDet][11-12 18:01:45]INFO:Train|Epoch:41/300|Iter:53902(23/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1414| loss_bbox:0.2571| loss_dfl:0.1563| aux_loss_qfl:0.1271| aux_loss_bbox:0.2271| aux_loss_dfl:0.1511|
INFO:NanoDet:Train|Epoch:41/300|Iter:53902(23/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1414| loss_bbox:0.2571| loss_dfl:0.1563| aux_loss_qfl:0.1271| aux_loss_bbox:0.2271| aux_loss_dfl:0.1511|
[NanoDet][11-12 18:01:46]INFO:Train|Epoch:41/300|Iter:53904(25/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1201| loss_bbox:0.2239| loss_dfl:0.1444| aux_loss_qfl:0.1012| aux_loss_bbox:0.1900| aux_loss_dfl:0.1384|
INFO:NanoDet:Train|Epoch:41/300|Iter:53904(25/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1201| loss_bbox:0.2239| loss_dfl:0.1444| aux_loss_qfl:0.1012| aux_loss_bbox:0.1900| aux_loss_dfl:0.1384|
[NanoDet][11-12 18:01:46]INFO:Train|Epoch:41/300|Iter:53906(27/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1202| loss_bbox:0.2033| loss_dfl:0.1462| aux_loss_qfl:0.0932| aux_loss_bbox:0.1777| aux_loss_dfl:0.1412|
INFO:NanoDet:Train|Epoch:41/300|Iter:53906(27/1347)| mem: 9G| lr:9.59e-04| loss_qfl:0.1202| loss_bbox:0.2033| loss_dfl:0.1462| aux_loss_qfl:0.0932| aux_loss_bbox:0.1777| aux_loss_dfl:0.1412|