
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

# 이미지 정규화
def get_id_emb_from_image(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std
    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature

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
    id_img = cv2.imread(args.source_img_path)
    
    # landmark가 소스 이미지에서 얼굴을 인식하지 못하였을 경우 디텍트 에러
    landmark = landmarkModel.get(id_img)
    if landmark is None:
        print('**** No Face Detect Error ****')
        exit()
    
    # 디텍션 된 얼굴을 수평으로 맞춤
    source_aligned_images = source_face_align(landmarkModel, args.source_img_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture()
    cap.open(args.target_video_path)
    videoWriter = cv2.VideoWriter(os.path.join(args.output_path, os.path.basename(args.target_video_path)), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in tqdm(range(int(all_f))):
        ret, frame = cap.read()
        
        # 타겟 이미지 얼굴 디텍트
        # target_aligned_images = target_face_align(landmarkModel, frame, args.image_size)
        target_aligned_images = target_face_align(landmarkModel,i, frame, args.target_video_path, args.image_size)
        if target_aligned_images == False:
            print('**** No Face Detect Error ****') 
        for idx, target_aligned_image in enumerate(target_aligned_images):
            # ArcFace.pdparams 얼굴 인식 알고리즘) 얼굴의 특징을 뽑아냄
            id_emb, id_feature = get_id_emb_from_image(id_net, source_aligned_images[idx % len(source_aligned_images)][0])
            # FaceSwap 모델 Load (UNET 베이스)
            faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
            faceswap_model.eval()
            
            # OpenCV에서 로드한 이미지를 PaddlePaddle에서도 사용가능하게 만듦
            att_img = cv2paddle(target_aligned_image[0])
            
            # 결과 이미지와 마스크 추출(얼굴 영역) <-- 타겟 이미지 학습 진행
            res, mask = faceswap_model(att_img)
            
            # Paddle to OpenCV
            res = paddle2cv(res)
            
            back_matrix = target_aligned_images[idx % len(target_aligned_images)][1]
            mask = np.transpose(mask[0].numpy(), (1, 2, 0))
            
            # back_matrix를 가지고 다시 역으로 합성을 진행함
            res = dealign(res, frame, back_matrix, mask)
            frame = res
        videoWriter.write(frame)
    cap.release()
    videoWriter.release()

# 소스 이미지 수평 맞춤
def source_face_align(landmarkModel, image_path, image_size=224):
    aligned_imgs =[]
    if os.path.isfile(image_path):
        img_list = [image_path]
    else:
        img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    for path in img_list:
        img = cv2.imread(path)
        landmarks = landmarkModel.get(img)
        if landmarks is not None:
            aligned_img, back_matrix = align_img(img, landmarks, image_size)
            aligned_imgs.append([aligned_img, back_matrix])
    return aligned_imgs
    
# 타겟 이미지 수평 맞춤
def target_face_align(landmarkModel,frame, image, target_video_path, image_size=224):
    aligned_imgs =[]
    
    video_name = target_video_path.split('/')[-1].replace('.mp4', '').replace('.avi', '').replace('.mpg', '').replace('.mpeg', '').replace('.mov', '')
    frame_folder = f"asset/{video_name}_target/{video_name}_target_{frame}"
    os.makedirs(frame_folder, exist_ok=True)
    
    landmarks = landmarkModel.gets(image)
    for target_count, landmark in enumerate(landmarks):
        if landmark is not None:
            # 디텍션 한 얼굴을 수평으로 맞춤과 동시에 다시 되돌리기 위한 back_matrix 계산
            aligned_img, back_matrix = align_img(image, landmark, image_size)
            aligned_imgs.append([aligned_img, back_matrix])
            
            cv2.imwrite(frame_folder + f'/{video_name}_target_{target_count}.png', aligned_img)
            np.save(frame_folder + f'/{video_name}_target_back_{target_count}.npy', back_matrix)
        
    return aligned_imgs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")

    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--source_img_path', type=str, help='path to the source image')
    parser.add_argument('--target_video_path', type=str, help='path to the target video')
    parser.add_argument('--output_path', type=str, default='results', help='path to the output videos')
    parser.add_argument('--image_size', type=int, default=224,help='size of the test images (224 SimSwap | 256 FaceShifter)')
    parser.add_argument('--merge_result', type=bool, default=True, help='output with whole image')
    parser.add_argument('--use_gpu', type=bool, default=False)

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    video_test_multi_face(args)