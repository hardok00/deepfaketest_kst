import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm

from utils.face_analysis import FaceAnalysis

def face_blur(args, landmark_2d):
    
    verify_landmark =[]
    verify_list = [os.path.join(args.verify_img_path, x) for x in os.listdir(args.verify_img_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    print(verify_list)
    for path in verify_list:
        verify_img = cv2.imread(path)
        bbox1, landmark = landmark_2d.get(verify_img)
        if bbox1 is None:
            print(f"***************{path} Face No Detect***************")
        else:
            verify_landmark.append(landmark)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture()
    cap.open(args.target_video_path)
    videoWriter = cv2.VideoWriter(os.path.join(args.output_dir, os.path.basename(args.target_video_path)), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
    for i in tqdm(range(int(all_f))):
        ret, frame = cap.read()
        bboxes, faces = landmark_2d.gets(frame,verify_landmark,verify_list)
        if bboxes is None:
            print("***************Target Face No Detect***************")
        else:
            tim = frame.copy()
            height, width, _ = frame.shape
            color = (200, 160, 75)
            for face in faces:
                lmk = face.landmark_2d_106
                lmk = np.round(lmk).astype(np.int64)
                convexhull = cv2.convexHull(lmk)
                
                mask = np.zeros((height, width), np.uint8)
                cv2.fillConvexPoly(mask, convexhull, 255)
                
                tim = cv2.blur(tim, (27,27))
                face_extracted = cv2.bitwise_and(tim, tim, mask=mask)
                
                background_mask = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(frame, frame, mask=background_mask)
                
                frame = cv2.add(background, face_extracted)
            videoWriter.write(frame)
    cap.release()
    videoWriter.release()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--verify_img_path', type=str, help='path to the source image')
    parser.add_argument('--target_video_path', type=str, help='path to the target images')
    parser.add_argument('--output_dir', type=str, default='poly_face_blur', help='path to the output dirs')
    parser.add_argument('--need_align', type=bool, default=True, help='need to align the image')
    
    args = parser.parse_args()
    if args.need_align:
        landmark_2d = FaceAnalysis(name='landmarks')
        landmark_2d.prepare(ctx_id=0, det_thresh = 0.6, det_size=(640, 640))
    os.makedirs(args.output_dir, exist_ok=True)
    face_blur(args, landmark_2d)