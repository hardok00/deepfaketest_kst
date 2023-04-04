# insightface version 0.2.1 => 0.7.3 변경 후 코드
# verify로 올린 비교 사진과 target 사진을 비교하여 0.5 이상인 target을 source 얼굴로 변경

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
import json

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

def image_test_multi_face(args, landmarkModel):
    if os.path.isfile(args.json_path):
        with open(args.json_path, 'rt', encoding='UTF-8') as annotations:
            coco = json.load(annotations)
    
    if os.path.isfile(args.target_img_path):
        img_list = [args.target_img_path]
    else:
        img_list = [os.path.join(args.target_img_path, x) for x in os.listdir(args.target_img_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    img_list.sort()
    
    for idx, path in enumerate(img_list):
        start_idx = path.rfind('/')
        if start_idx > 0:
            target_name = path[path.rfind('/'):]
        else:
            target_name = args.target_img_path

        origin_att_img = cv2.imread(path)
        h,w,_ =  origin_att_img.shape
        bboxes = []

        for image_id in coco['annotations']:
            if image_id["image_id"] == idx + 1 and image_id["category_id"] == 2:
                image_box = image_id["bbox"]
                image_box = list(map(int, image_box))
                print(image_box)
                cropped_image = origin_att_img[image_box[1]:image_box[1]+image_box[3], image_box[0]:image_box[0]+image_box[2]]
                
                bbox, landmarks = landmarkModel.gets(cropped_image)
                print(f"crop bbox : {bbox}" )
                bbox1, landmarks = landmarkModel.gets(origin_att_img)
                
                for j in bbox:
                    j = list(map(int, j))
                    del j[4]
                    # print(j)
                    bboxes.append([image_box[0]+j[0],image_box[1]+j[1],image_box[0]+j[2],image_box[1]+j[3]])
                print(bboxes)
                    
                print(f"image bbox : {bbox1}" )
                    
                # print(bboxes)
                # cv2.imwrite(os.path.join(args.output_dir, os.path.basename(target_name)), cropped_image)
        
        for idx, bbox in enumerate(bboxes):
            # print(bbox)
            if bbox[1] < 0:
                bbox[1] = 0
            p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            origin_att_img[p1[1]: p2[1], p1[0]:p2[0], :] = cv2.GaussianBlur(origin_att_img[p1[1]: p2[1], p1[0]:p2[0], :], (0,0), 20 )
            # draw_text(origin_att_img, str(bbox[4]), pos=(int(bbox[0]), int(bbox[1])), font_scale=1, font_thickness= 1, text_color=(255, 255, 255), text_color_bg=(0, 0, 255))
        
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(target_name).format(idx)), origin_att_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--target_img_path', type=str, help='path to the target images')
    parser.add_argument('--json_path', type=str, help='path to the json')
    parser.add_argument('--output_dir', type=str, default='results', help='path to the output dirs')
    parser.add_argument('--image_size', type=int, default=224,help='size of the test images (224 SimSwap | 256 FaceShifter)')
    parser.add_argument('--merge_result', type=bool, default=True, help='output with whole image')
    parser.add_argument('--need_align', type=bool, default=True, help='need to align the image')
    parser.add_argument('--use_gpu', type=bool, default=False)


    args = parser.parse_args()
    if args.need_align:
        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
        # bboxes = target_faces_align(landmarkModel, args.target_img_path, args.json_path, args.image_size)
    os.makedirs(args.output_dir, exist_ok=True)
    image_test_multi_face(args, landmarkModel)












