
from __future__ import division

import os
import glob
import os.path as osp
import cv2

import numpy as np
import onnxruntime
from numpy.linalg import norm

from insightface.model_zoo import model_zoo
from insightface.app.common import Face

__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, name, root='./checkpoints', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                continue
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                # del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']
        self.rec_model = self.models['recognition']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return bboxes, ret

    def gets(self, target_img, verify_landmark, verify_list, max_num=0):
        # bboxes1, kpss1 = self.det_model.detect(target_img, max_num=max_num, metric='default')
        
        bboxes1, kpss1 = self.get(target_img)
        
        kps_img = kpss1
        kps_bbox = bboxes1
        nt_img = []
        
        for idx, kpss2 in enumerate(verify_landmark):
            verify_img = cv2.imread(verify_list[idx])
            print(kpss2[0]['kps'])
            # print(kpss2[idx]['kps'])
            
            for j in range(bboxes1.shape[0]):
                kps1 = face(kpss1[j]['kps'])
                print(f"kps1 : {kps1.kps}")
                feat1 = self.rec_model.get(target_img, kps1)

                kps2 = face(kpss2[0]['kps'])
                feat2 = self.rec_model.get(verify_img, kps2)
                
                sim = self.rec_model.compute_sim(feat1, feat2)

                print(f"일치 확률 : {sim}")
                
                # if sim < 0.4:
                #     kps_img.append(kpss1[j])
                #     kps_bbox.append(bboxes1[j])
                    
                if sim >= 0.4:
                    nt_img.append(j)
                    break
                
        if nt_img:
            nt_img.sort(reverse=True)
            for delete in nt_img:
                del kps_img[delete]
                
        return kps_bbox, kps_img
    
    
    def j_gets(self, target_img, verify_landmark, verify_list, max_num=0):
        # bboxes1, kpss1 = self.det_model.detect(target_img, max_num=max_num, metric='default')
        bboxes1, kpss1 = self.get(target_img)
        
        kps_img = []
        kps_bbox = []
        nt_img = []
        
        for idx, kpss2 in enumerate(verify_landmark):
            verify_img = cv2.imread(verify_list[idx])
            print(kpss2[0]['kps'])
            # print(kpss2[idx]['kps'])
            
            for j in range(bboxes1.shape[0]):
                if j in nt_img:
                    continue
                kps1 = face(kpss1[j]['kps'])
                print(f"kps1 : {kps1.kps}")
                feat1 = self.rec_model.get(target_img, kps1)

                kps2 = face(kpss2[0]['kps'])
                feat2 = self.rec_model.get(verify_img, kps2)
                
                sim = self.rec_model.compute_sim(feat1, feat2)

                print(f"일치 확률 : {sim}")
                
                if sim < 0.4:
                    kps_img.append(kpss1[j])
                    kps_bbox.append(bboxes1[j])
                    
                if sim >= 0.4:
                    nt_img.append(j)
                    kps_img = []
                    kps_bbox = []
            
                
        return kps_bbox, kps_img

class face():
    def __init__(self, kps):
        self.kps = kps
