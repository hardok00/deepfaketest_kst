import cv2
import numpy as np
import argparse
import os

from utils.face_analysis import FaceAnalysis

def face_blur(args, landmark_2d):
    start_idx = args.target_img_path.rfind('/')
    if start_idx > 0:
        target_name = args.target_img_path[args.target_img_path.rfind('/'):]
    else:
        target_name = args.target_img_path
        
    img = cv2.imread(args.target_img_path)
    bboxes, faces = landmark_2d.get(img)
    #assert len(faces)==6
    tim = img.copy()
    height, width, _ = img.shape
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
        background = cv2.bitwise_and(img, img, mask=background_mask)
        
        img = cv2.add(background, face_extracted)
        
        # lmk = np.round(lmk).astype(np.int64)
        for i in range(lmk.shape[0]):
            p = tuple(lmk[i])
            cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.output_dir, os.path.basename(target_name)), img)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--target_img_path', type=str, help='path to the target images')
    parser.add_argument('--output_dir', type=str, default='poly_face_blur', help='path to the output dirs')
    parser.add_argument('--need_align', type=bool, default=True, help='need to align the image')
    
    args = parser.parse_args()
    if args.need_align:
        landmark_2d = FaceAnalysis(name='landmarks', root='./checkpoints', allowed_modules=['detection', 'landmark_2d_106'])
        landmark_2d.prepare(ctx_id=0, det_thresh = 0.6, det_size=(640, 640))
    os.makedirs(args.output_dir, exist_ok=True)
    face_blur(args, landmark_2d)