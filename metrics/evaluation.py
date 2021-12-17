import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn import metrics

class Evaluation:

    def convert2mask(self, mt, shape):
        # Converts coordinates of bounding-boxes into blank matrix with values set where bounding-boxes are.

        t = np.zeros([shape, shape])
        for m in mt:
            x, y, w, h = m
            cv2.rectangle(t, (x,y), (x+w, y+h), 1, -1)
        return t

    def prepare_for_detection(self, prediction, ground_truth):
            # For the detection task, convert Bounding-boxes to masked matrices (0 for background, 1 for the target). If you run segmentation, do not run this function

            if len(prediction) == 0:
                return [], []

            # Large enough size for base mask matrices:
            shape = 2*max(np.max(prediction), np.max(ground_truth)) 
            
            p = self.convert2mask(prediction, shape)
            gt = self.convert2mask(ground_truth, shape)

            return p, gt

    def iou_compute(self, p, gt):
            # Computes Intersection Over Union (IOU)
            if len(p) == 0:
                return 0
            intersection = np.logical_and(p, gt)
            union = np.logical_or(p, gt)

            iou = np.sum(intersection) / np.sum(union)

            return iou

    # Add your own metrics here, such as mAP, class-weighted accuracy, ...
    def prec_recall(self, iou_list, n_ground_truth=250, name=""):
        ps, rs = [],[]
        for i in range(11):
            limit = i/10
            tp, fp, fn = 0, 0, 0
            for iou in iou_list:
                if iou >= limit: tp += 1
                elif iou < limit and iou != 0: fp += 1
                else: fn += 1
            p, r = tp/(tp+fp), tp/n_ground_truth
            ps.append(p)
            rs.append(r)
        areaPR = metrics.auc(rs, ps)
        plt.plot(rs,ps)
        plt.title(f"PR curve for {name}, area: 0.{int(areaPR*1000)}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()
        print(areaPR)
        return ps, rs