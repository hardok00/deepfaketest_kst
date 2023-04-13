
import paddle
import argparse
import cv2
import numpy as np
import os
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from tqdm import tqdm

# 신뢰도 점수 이미지에 출력
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def video_test_multi_face(args):
    # GPU or CPU 셋팅
    paddle.set_device("gpu" if args.use_gpu else 'cpu')
    
    # 모델 Load
    faceswap_model = FaceSwap(args.use_gpu)

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))

    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')

    # 얼굴 탐지를 위해 LandmarkModel ONNX LOAD
    landmarkModel = LandmarkModel(name='landmarks')
    # det_thresh = 예측 한계치 설정
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture()
    cap.open(args.target_video_path)
    videoWriter = cv2.VideoWriter(os.path.join(args.output_path, os.path.basename(args.target_video_path)), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in tqdm(range(int(all_f))):
        ret, frame = cap.read()
        
        # 타겟 이미지 얼굴 디텍트
        bboxes, landmarks = landmarkModel.gets(frame)
        if bboxes is None:
            print("***************Target Face No Detect***************")
        else:
            for bbox in bboxes:
                if bbox[1] < 0:
                    bbox[1] = 0
                # print(bbox)
                p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                frame[p1[1]: p2[1], p1[0]:p2[0], :] = cv2.GaussianBlur(frame[p1[1]: p2[1], p1[0]:p2[0], :], (0,0), 20 )
            videoWriter.write(frame)
    cap.release()
    videoWriter.release()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")

    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--target_video_path', type=str, help='path to the target video')
    parser.add_argument('--output_path', type=str, default='results', help='path to the output videos')
    parser.add_argument('--image_size', type=int, default=224,help='size of the test images (224 SimSwap | 256 FaceShifter)')
    parser.add_argument('--merge_result', type=bool, default=True, help='output with whole image')
    parser.add_argument('--use_gpu', type=bool, default=False)

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    video_test_multi_face(args)