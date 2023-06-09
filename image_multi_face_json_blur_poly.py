import cv2
import numpy as np
import argparse
import os
import json

from utils.face_analysis import FaceAnalysis

def face_blur(args, landmark_2d):
    if os.path.isfile(args.json_path):
        with open(args.json_path, 'rt', encoding='UTF-8') as annotations:
            coco = json.load(annotations)
    else:
        print("********************* JSON FILE NOT DETECT *****************************")
        return None
    
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
        if bboxes is None:
            print(f"***************{path} Face No Detect***************")
        else:
            verify_landmark.append(landmark)
        
    for idx, path in enumerate(img_list):
        print(path)
        start_idx = path.rfind('/')
        if start_idx > 0:
            target_name = path[path.rfind('/'):]
        else:
            target_name = args.target_img_path

        target_img = cv2.imread(path)
        
        for image_id in coco['annotations']:
            if image_id["image_id"] == idx + 1 and image_id["category_id"] == 2:
                image_box = image_id["bbox"]
                image_box = list(map(int, image_box))
                
                cropped_image = target_img[image_box[1]:image_box[1]+image_box[3], image_box[0]:image_box[0]+image_box[2]]
            
                bboxes2, faces = landmark_2d.j_gets(cropped_image, verify_landmark, verify_list)
                if bboxes2 is None:
                    print("***************Cropped Face No Detect***************")
                else:
                    tim = target_img.copy()
                    height, width, _ = target_img.shape
                    for face in faces:
                        lmk = face.landmark_2d_106
                        for row in lmk:
                            row[0] += image_box[0]
                            if row[0] < 0:
                                row[0] = 0
                            elif row[0] > width:
                                row[0] = width
                            row[1] += image_box[1]
                            if row[1] < 0:
                                row[1] = 0
                            elif row[1] > height:
                                row[1] = height
                        lmk = np.round(lmk).astype(np.int64)
                        convexhull = cv2.convexHull(lmk)
                        
                        mask = np.zeros((height, width), np.uint8)
                        cv2.fillConvexPoly(mask, convexhull, 255)
                        
                        tim = cv2.blur(tim, (27,27))
                        face_extracted = cv2.bitwise_and(tim, tim, mask=mask)
                        
                        background_mask = cv2.bitwise_not(mask)
                        background = cv2.bitwise_and(target_img, target_img, mask=background_mask)
                        
                        target_img = cv2.add(background, face_extracted)
                    
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(target_name)), target_img)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--verify_img_path', type=str, help='path to the verify image')
    parser.add_argument('--target_img_path', type=str, help='path to the target images')
    parser.add_argument('--json_path', type=str, help='path to the json')
    parser.add_argument('--output_dir', type=str, default='poly_face_blur', help='path to the output dirs')
    parser.add_argument('--image_size', type=int, default=224,help='size of the test images (224 SimSwap | 256 FaceShifter)')
    parser.add_argument('--merge_result', type=bool, default=True, help='output with whole image')
    parser.add_argument('--need_align', type=bool, default=True, help='need to align the image')
    parser.add_argument('--use_gpu', type=bool, default=False)
    
    args = parser.parse_args()
    if args.need_align:
        landmark_2d = FaceAnalysis(name='landmarks')
        landmark_2d.prepare(ctx_id=0, det_thresh = 0.6, det_size=(640, 640))
    os.makedirs(args.output_dir, exist_ok=True)
    face_blur(args, landmark_2d)
    
    