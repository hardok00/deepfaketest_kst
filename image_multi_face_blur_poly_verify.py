import cv2
import numpy as np
import argparse
import os
import json

from utils.face_analysis import FaceAnalysis
from utils.prepare_data_json_verify import LandmarkModel

def face_blur(args, landmark_2d):
    # if os.path.isfile(args.json_path):
    #     with open(args.json_path, 'rt', encoding='UTF-8') as annotations:
    #         coco = json.load(annotations)
    # else:
    #     print("********************* JSON FILE NOT DETECT *****************************")
    #     return None
    
    start_idx = args.target_img_path.rfind('/')
    if start_idx > 0:
        target_name = args.target_img_path[args.target_img_path.rfind('/'):]
    else:
        target_name = args.target_img_path
    
    if os.path.isfile(args.target_img_path):
        img_list = [args.target_img_path]
    else:
        img_list = [os.path.join(args.target_img_path, x) for x in os.listdir(args.target_img_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    img_list.sort()
    
    verify_landmark =[]
    verify_list = [os.path.join(args.verify_img_path, x) for x in os.listdir(args.verify_img_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    print(verify_list)
    for path in verify_list:
        verify_img = cv2.imread(path)
        bboxes ,landmark = landmark_2d.get(verify_img)
        
        verify_landmark.append(landmark)
    
    target_img = cv2.imread(args.target_img_path)
    bboxes2, faces = landmark_2d.gets(target_img, verify_landmark, verify_list)
    #assert len(faces)==6
    tim = target_img.copy()
    height, width, _ = target_img.shape
    # color = (200, 160, 75)
    for face in faces:
        lmk = face.landmark_2d_106
        lmk = np.round(lmk).astype(np.int64)
        convexhull = cv2.convexHull(lmk)
        
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, convexhull, 255)
        
        tim = cv2.blur(tim, (27,27))
        face_extracted = cv2.bitwise_and(tim, tim, mask=mask)
        
        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(target_img, target_img, mask=background_mask)
        
        target_img = cv2.add(background, face_extracted)
        
        # lmk = np.round(lmk).astype(np.int64)
        # for i in range(lmk.shape[0]):
        #     p = tuple(lmk[i])
        #     # cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.output_dir, os.path.basename(target_name)), target_img)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--verify_img_path', type=str, help='path to the verify image')
    parser.add_argument('--target_img_path', type=str, help='path to the target images')
    # parser.add_argument('--json_path', type=str, help='path to the json')
    parser.add_argument('--output_dir', type=str, default='poly_face_blur', help='path to the output dirs')
    parser.add_argument('--image_size', type=int, default=224,help='size of the test images (224 SimSwap | 256 FaceShifter)')
    parser.add_argument('--merge_result', type=bool, default=True, help='output with whole image')
    parser.add_argument('--need_align', type=bool, default=True, help='need to align the image')
    parser.add_argument('--use_gpu', type=bool, default=False)
    
    args = parser.parse_args()
    if args.need_align:
        landmark_2d = FaceAnalysis(name='landmarks')
        # , allowed_modules=['detection', 'landmark_2d_106']
        landmark_2d.prepare(ctx_id=0, det_thresh = 0.6, det_size=(640, 640))
    os.makedirs(args.output_dir, exist_ok=True)
    face_blur(args, landmark_2d)
    
    
