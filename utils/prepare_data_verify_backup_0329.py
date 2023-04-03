# insightface version 0.2.1 => 0.7.3 변경 후 코드
# verify로 올린 비교 사진과 target 사진을 비교하여 0.5 이상인 target을 source 얼굴로 변경

import os
import cv2
import numpy as np
import glob
import os.path as osp
from insightface.model_zoo import model_zoo

# ONNX 모델
class LandmarkModel():
    def __init__(self, name, root='./checkpoints'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        assert 'recognition' in self.models
        self.det_model = self.models['detection']
        self.rec_model = self.models['recognition']

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            print(model)
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    """
    bboxes : 이미지에서 검출한 얼굴 영역의 위치와 크기를 나타내는 변수
    열 0 : 얼굴 영역의 x 좌표
    열 1 : 얼굴 영역의 y 좌표
    열 2 : 얼굴 영역의 너비
    열 3 : 얼굴 영역의 높이
    열 4 : 얼굴 영역의 신뢰도 점수 (detection score)
    
    kpss : 검출된 얼굴 영역의 특징점 위치를 나타내는 2차원 좌표 배열
    """
    # max_num 0 = 모든 객체에 대해 탐지
    def get(self, img, max_num=0):
        # bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return None
        
        det_score = bboxes[..., 4]

        # 탐지 점수가 가장 높은 얼굴 선택
        best_index = np.argmax(det_score)

        kps = None
        if kpss is not None:
            kps = kpss[best_index]
        return kps

    def gets(self, target_img, verify_img, max_num=0):
        # bboxes, kpss = self.det_model.detect(target_img, threshold=self.det_thresh, max_num=max_num, metric='default')
        bboxes1, kpss1 = self.det_model.detect(target_img, max_num=max_num, metric='default')
        bboxes2, kpss2 = self.det_model.detect(verify_img, max_num=max_num, metric='default')

        kps_img = []
    
        print(kpss1.shape[0])
        
        for i in range(kpss1.shape[0]):
            kps1 = face(kpss1[i])
            print(f"kps1 : {kps1.kps}")
            feat1 = self.rec_model.get(target_img, kps1)

            kps2 = face(kpss2[0])
            feat2 = self.rec_model.get(verify_img, kps2)
            
            sim = self.rec_model.compute_sim(feat1, feat2)

            print(sim)
            
            if sim > 0.4:
                kps_img.append(kpss1[i])
                break
            
        # 특정 행을 삭제하고 싶을 때
        # bboxes = np.delete(bboxes, 1, axis= 0)
        # kpss = np.delete(kpss, 1, axis= 0)
        
        return bboxes1 ,kps_img
    
class face():
    def __init__(self, kps):
        self.kps = kps