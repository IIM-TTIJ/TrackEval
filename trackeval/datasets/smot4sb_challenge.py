import os
import numpy as np
from copy import deepcopy
from trackeval.utils import TrackEvalException
from trackeval.datasets._BaseDataset import _BaseDataset

class SMOT4SBDataset(_BaseDataset):
    def __init__(self, config=None):
        super().__init__()
        self.config = self.get_default_dataset_config()
        if config is not None:
            self.config.update(config)
        
        self.tracker_list = []
        self.seq_list = []
        self.class_list = ['pedestrian']
        self.output_fol = self.config['OUTPUT_FOLDER']
        self.output_sub_fol = "mot_results"
        self.should_classes_combine = False
        self.use_super_categories = False

    @staticmethod
    def get_default_dataset_config():
        return {
            'GT_FOLDER': '',
            'TRACKERS_FOLDER': '',
            'OUTPUT_FOLDER': '',
            'CLASSES_TO_EVAL': ['pedestrian'],
            'MIN_IOU_THRESHOLD': 0.5
        }

    def _load_raw_file(self, tracker, seq, is_gt):
        folder = self.config['GT_FOLDER'] if is_gt else self.config['TRACKERS_FOLDER']
        file_path = os.path.join(folder, seq, tracker + ".txt")
        if not os.path.isfile(file_path):
            raise TrackEvalException(f"File {file_path} not found.")

        data = np.loadtxt(file_path, delimiter=',')
        return {
            'ids': data[:, 1].astype(int),
            'dets': data[:, 2:6],
            'confidences': data[:, 6] if not is_gt else None
        }

    def get_preprocessed_seq_data(self, raw_data, cls):
        return {
            'ids': deepcopy(raw_data['ids']),
            'dets': deepcopy(raw_data['dets']),
            'confidences': deepcopy(raw_data['confidences']) if raw_data['confidences'] is not None else None
        }

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        return self._calculate_box_ious(gt_dets_t, tracker_dets_t)