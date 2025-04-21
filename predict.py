import cv2
import json
import torch
from torchvision import transforms

from Model.YOLOV8 import YOLOv8_face
from Model.MTFSNet import FERModel
from Utiles.util import LoadParameter , set_seed


def draw_img(box,emo_label,srcimg):
    x, y, w, h = box.astype(int)
    colors = {
        "Happy": (0, 255, 0),    # 绿色
        "Interest": (0, 0, 255), # 红色
        "Confuse": (255, 0, 0),  # 蓝色
        "Bored": (255, 255, 0),  # 黄色
        "Natural": (0, 255, 255) # 青色
    }

    # 根据情绪标签选择颜色
    color = colors.get(emo_label, (255, 255, 255))  # 默认为白色
    cv2.rectangle(srcimg, (x, y), (x + w, y + h), color, thickness=1)
    cv2.putText(srcimg, "Emo: "+emo_label , (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color,thickness=1)
    return srcimg


if __name__ == "__main__":
    parameterRead=None
    with open('config.json') as f:
        parameterRead = json.load(f)

    Yolodefine=parameterRead['Yolov8']
    MFSTdefine=parameterRead['MFSTNet']
    video_source=parameterRead['Video_path']
    Result_define=parameterRead['Result_info']
    device_id=parameterRead['DEVICE']
    data_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((MFSTdefine['img_height'], MFSTdefine['img_wide'])),
                                    transforms.Normalize(mean=MFSTdefine['mean'],
                                    std=MFSTdefine['std'])
                                ])
    
    DEVICE=torch.device(device_id)
    set_seed(MFSTdefine["seed"])
    YOLOv8_face_detector = YOLOv8_face(Yolodefine['modelpath'], conf_thres=Yolodefine['confThreshold'], iou_thres=Yolodefine['nmsThreshold'])
    MTFsNet_Emo_detector=FERModel(labels=MFSTdefine["label_self"],num_class=MFSTdefine["num_class"],data_transforms=data_transforms,device_id=device_id)
    MTFsNet_Emo_detector.model = LoadParameter(MTFsNet_Emo_detector.model,MFSTdefine["modelpath"],device=device_id)
    left_tmp_frames=[]
    right_tmp_frames=[]
    buffer_sizes=MFSTdefine["forzen_frame"]

    cap=cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
    if Result_define["Save_Result"]:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
        out = cv2.VideoWriter(Result_define['Save_Path'], fourcc, fps, (width, height))  # 写入视频
    time_pause=5
    tmp=-1
    count=0
    while(True):
        ret,srcimg=cap.read()

        if ret :
            tmp+=1
            count+=1
            if tmp%time_pause==0:
                tmp=0
                boxes, scores, classids, kpts = YOLOv8_face_detector.detect(srcimg)
            for box in boxes:
                #右侧人物:
                Emo="Natural"
                x, y, w, h = box.astype(int)
                right_tmp_frames.append(data_transforms(srcimg[y:y+h,x:x+w,:]))
                if len(right_tmp_frames)==buffer_sizes:
                    Emo=MTFsNet_Emo_detector.fer(right_tmp_frames)
                    right_tmp_frames.pop(0)
                    srcimg=draw_img(box,Emo,srcimg)
            if Result_define["Save_Result"]:
                out.write(srcimg)  # 写入帧
            if Result_define['Show_Result']:  
                cv2.namedWindow("frame",0)
                cv2.resizeWindow("frame", 1080, 1080)
                cv2.imshow('frame',srcimg)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if count%fps ==0:
                print(str(count/fps)+" has been processed!")
        else:
            break
    cap.release()
    if Result_define['Show_Result']:
        cv2.destroyAllWindows()
    if Result_define['Save_Result']:
        out.release()
