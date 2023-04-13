
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

def get_id_emb_from_image(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std
    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature

def image_test_multi_face(args, source_aligned_images, target_aligned_images, bboxes):
    paddle.set_device("gpu" if args.use_gpu else 'cpu')
    faceswap_model = FaceSwap(args.use_gpu)

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))

    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')

    start_idx = args.target_img_path.rfind('/')
    if start_idx > 0:
        target_name = args.target_img_path[args.target_img_path.rfind('/'):]
    else:
        target_name = args.target_img_path
    origin_att_img = cv2.imread(args.target_img_path)

    for idx, target_aligned_image in enumerate(target_aligned_images):
        id_emb, id_feature = get_id_emb_from_image(id_net, source_aligned_images[idx % len(source_aligned_images)][0])
        faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
        faceswap_model.eval()

        att_img = cv2paddle(target_aligned_image[0])

        res, mask = faceswap_model(att_img)
        res = paddle2cv(res)

        back_matrix = target_aligned_images[idx % len(target_aligned_images)][1]
        mask = np.transpose(mask[0].numpy(), (1, 2, 0))
        origin_att_img = dealign(res, origin_att_img, back_matrix, mask)
        
    for bbox in bboxes:
        if bbox[1] < 0:
            bbox[1] = 0
        # print(bbox)
        draw_text(origin_att_img, str(bbox[4]), pos=(int(bbox[0]), int(bbox[1])), font_scale=1, font_thickness= 1, text_color=(255, 255, 255), text_color_bg=(0, 0, 255))
    
    cv2.imwrite(os.path.join(args.output_dir, os.path.basename(target_name.format(idx))), origin_att_img)

# 소스이미지 등록 및 추출
def face_align(landmarkModel, image_path, image_size=224):
    aligned_imgs =[]
    if os.path.isfile(image_path):
        img_list = [image_path]
    else:
        img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    for path in img_list:
        img = cv2.imread(path)
        bboxes, landmark = landmarkModel.get(img)
        if landmark is not None:
            # base_path = path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
            aligned_img, back_matrix = align_img(img, landmark, image_size)
            aligned_imgs.append([aligned_img, back_matrix])
            
            # cv2.imwrite(base_path + '_aligned.png', aligned_img)
            # np.save(base_path + '_back.npy', back_matrix)
        else:
            print("*************source face no detect***************")
            
    return aligned_imgs

def faces_align(landmarkModel, image_path, image_size=224):
    aligned_imgs =[]
    target_count = 0
    if os.path.isfile(image_path):
        img_list = [image_path]
    else:
        img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    for path in img_list:
        img = cv2.imread(path)
        name = path.split('/')[-1].replace('.png', '').replace('.jpg', '').replace('.jpeg', '')

        target_path = f"asset/{name}_target"
        os.makedirs(target_path, exist_ok=True)
        
        bboxes, landmarks = landmarkModel.gets(img)
        if bboxes is None:
            print(f"***************Target Face No Detect***************")
        
        # print(landmarks)
        
        for target_count, landmark in enumerate(landmarks):
            if landmark is not None:
                aligned_img, back_matrix = align_img(img, landmark, image_size)
                aligned_imgs.append([aligned_img, back_matrix])
                                            
                # 타겟 이미지 여러개 추출
                cv2.imwrite(target_path + f'/{name}_target_{target_count}.png', aligned_img)
                np.save(target_path + f'/{name}_target_back_{target_count}.npy', back_matrix)

        target_list = [os.path.join(target_path, x) for x in os.listdir(target_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
        
        print(target_list)
    return bboxes,aligned_imgs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--source_img_path', type=str, help='path to the source image')
    parser.add_argument('--target_img_path', type=str, help='path to the target images')
    parser.add_argument('--output_dir', type=str, default='results', help='path to the output dirs')
    parser.add_argument('--image_size', type=int, default=224,help='size of the test images (224 SimSwap | 256 FaceShifter)')
    parser.add_argument('--merge_result', type=bool, default=True, help='output with whole image')
    parser.add_argument('--need_align', type=bool, default=True, help='need to align the image')
    parser.add_argument('--use_gpu', type=bool, default=False)


    args = parser.parse_args()
    if args.need_align:
        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
        source_aligned_images = face_align(landmarkModel, args.source_img_path)
        bboxes, target_aligned_images = faces_align(landmarkModel, args.target_img_path, args.image_size)
    os.makedirs(args.output_dir, exist_ok=True)
    image_test_multi_face(args, source_aligned_images, target_aligned_images, bboxes)






