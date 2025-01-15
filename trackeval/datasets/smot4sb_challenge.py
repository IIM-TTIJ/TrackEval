import os
import csv
import configparser
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .mot_challenge_2d_box import MotChallenge2DBox
from .. import utils
from .. import _timing
from ..utils import TrackEvalException


class SMOT4SBChallenge(MotChallenge2DBox):
    """Dataset class for MOT Challenge 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
            'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
            'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # '{gt_folder}/{seq}/gt/gt.txt'
            'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
                                      # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
                                      # If True, then the middle 'benchmark-split' folder is skipped for both.
            'USE_SO_HOTA': True, # SO-HOTA differs from conventional IoU-based tracking metrics because 
                                 # it is calculated based on DotD. Accordingly, basic metrics such as 
                                 # 'IDTP', 'IDFN', 'IDFP', 'Dets', 'IDs', etc. will also change. 
                                 # In order to separate the results computed in different ways, when using SO-HOTA, 
                                 # it is necessary not only to create the evaluation indicator class, 
                                 # but also to create a separate instance of this class after setting this flag.
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__(config)
        # Fill non-given config values with defaults

        # Get classes to eval
        # self.valid_classes = ['pedestrian']
        # self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
        #                    for cls in self.config['CLASSES_TO_EVAL']]
        # if not all(self.class_list):
        #     raise TrackEvalException('Attempted to evaluate an invalid class. Only pedestrian class is valid.')
        # self.class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
        #                                'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
        #                                'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}
        # self.valid_class_numbers = list(self.class_name_to_class_id.values())

        self.use_so_hota = self.config['USE_SO_HOTA']
        if self.use_so_hota:
            self.S = None  # Dataset-wide normalization factor for DotD
            self.compute_S_for_dataset()  # Caliculate normalization factor S for DotD
        
    @staticmethod
    def compute_S_for_dataset(self):
        """Compute the normalization factor S from ground truth bounding boxes across the entire dataset."""
        all_wh = []
        for seq in self.seq_list:
            raw_data = self._load_raw_file(None, seq, is_gt=True)  # Load GT data
            for dets in raw_data['gt_dets']:
                if len(dets) > 0:
                    all_wh.append(dets[:, 2:4])  # Collect width & height

        if not all_wh:
            raise TrackEvalException("No valid ground truth bounding boxes found for computing S.")

        all_wh = np.vstack(all_wh)  # Merge all detections across sequences
        avg_size = np.mean(all_wh[:, 0] * all_wh[:, 1])  # Mean area
        self.S = np.sqrt(avg_size)  # Compute S

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        MOT Challenge:
            In MOT Challenge, the 4 preproc steps are as follow:
                1) There is only one class (pedestrian) to be evaluated, but all other classes are used for preproc.
                2) Predictions are matched against all gt boxes (regardless of class), those matching with distractor
                    objects are removed.
                3) There is no crowd ignore regions.
                4) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.
        """
        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        distractor_class_names = ['bird', 'static_person', 'distractor', 'reflection']
        if self.benchmark == 'MOT20':
            distractor_class_names.append('non_mot_vehicle')
        distractor_classes = [self.class_name_to_class_id[x] for x in distractor_class_names]
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Get all data
            gt_ids = raw_data['gt_ids'][t]
            gt_dets = raw_data['gt_dets'][t]
            gt_classes = raw_data['gt_classes'][t]
            gt_zero_marked = raw_data['gt_extras'][t]['zero_marked']

            tracker_ids = raw_data['tracker_ids'][t]
            tracker_dets = raw_data['tracker_dets'][t]
            tracker_classes = raw_data['tracker_classes'][t]
            tracker_confidences = raw_data['tracker_confidences'][t]
            similarity_scores = raw_data['similarity_scores'][t]

            # Evaluation is ONLY valid for pedestrian class
            if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
                raise TrackEvalException(
                    'Evaluation is only valid for pedestrian class. Non pedestrian class (%i) found in sequence %s at '
                    'timestep %i.' % (np.max(tracker_classes), raw_data['seq'], t))

            # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
            # which are labeled as belonging to a distractor class.
            to_remove_tracker = np.array([], np.int)
            if self.do_preproc and self.benchmark != 'MOT15' and gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:

                # Check all classes are valid:
                invalid_classes = np.setdiff1d(np.unique(gt_classes), self.valid_class_numbers)
                if len(invalid_classes) > 0:
                    print(' '.join([str(x) for x in invalid_classes]))
                    raise(TrackEvalException('Attempting to evaluate using invalid gt classes. '
                                             'This warning only triggers if preprocessing is performed, '
                                             'e.g. not for MOT15 or where prepropressing is explicitly disabled. '
                                             'Please either check your gt data, or disable preprocessing. '
                                             'The following invalid classes were found in timestep ' + str(t) + ': ' +
                                             ' '.join([str(x) for x in invalid_classes])))

                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                to_remove_tracker = match_cols[is_distractor_class]

            # Apply preprocessing to remove all unwanted tracker dets.
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Remove gt detections marked as to remove (zero marked), and also remove gt detections not in pedestrian
            # class (not applicable for MOT15)
            if self.do_preproc and self.benchmark != 'MOT15':
                gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & \
                                  (np.equal(gt_classes, cls_id))
            else:
                # There are no classes for MOT15
                gt_to_keep_mask = np.not_equal(gt_zero_marked, 0)
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    @_timing.time
    def get_raw_seq_data(self, tracker, seq):
        """ Loads raw data (tracker and ground-truth) for a single tracker on a single sequence.
        Raw data includes all of the information needed for both preprocessing and evaluation, for all classes.
        A later function (get_processed_seq_data) will perform such preprocessing and extract relevant information for
        the evaluation of each class.

        This returns a dict which contains the fields:
        [num_timesteps]: integer
        [gt_ids, tracker_ids, gt_classes, tracker_classes, tracker_confidences]:
                                                                list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, tracker_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [similarity_scores]: list (for each timestep) of 2D NDArrays.
        [gt_extras]: dict (for each extra) of lists (for each timestep) of 1D NDArrays (for each det).

        gt_extras contains dataset specific information used for preprocessing such as occlusion and truncation levels.

        Note that similarities are extracted as part of the dataset and not the metric, because almost all metrics are
        independent of the exact method of calculating the similarity. However datasets are not (e.g. segmentation
        masks vs 2D boxes vs 3D boxes).
        We calculate the similarity before preprocessing because often both preprocessing and evaluation require it and
        we don't wish to calculate this twice.
        We calculate similarity between all gt and tracker classes (not just each class individually) to allow for
        calculation of metrics such as class confusion matrices. Typically the impact of this on performance is low.
        """
        # Load raw data.
        raw_gt_data = self._load_raw_file(tracker, seq, is_gt=True)
        raw_tracker_data = self._load_raw_file(tracker, seq, is_gt=False)
        raw_data = {**raw_tracker_data, **raw_gt_data}  # Merges dictionaries

        # Calculate similarities for each timestep.
        similarity_scores_iou = []
        similarity_scores_dotd = []
        for t, (gt_dets_t, tracker_dets_t) in enumerate(zip(raw_data['gt_dets'], raw_data['tracker_dets'])):
            if self.use_so_hota:
                dotds = self._calculate_similarities_dotd(gt_dets_t, tracker_dets_t)
                similarity_scores_dotd.append(dotds)
            else:
                ious = self._calculate_similarities(gt_dets_t, tracker_dets_t)
                similarity_scores_iou.append(ious)

        if self.use_so_hota:
            raw_data['similarity_scores'] = similarity_scores_dotd  # DotD based evaluation
        else:
            raw_data['similarity_scores'] = similarity_scores_iou  # IoU based evaluation
        return raw_data
        
    @staticmethod
    def _calculate_box_dot_distance(self, bboxes1, bboxes2, S, box_format='xywh'):
        """Calculates Dot Distance (DotD) between two sets of bounding boxes."""
        if self.S is None:
            raise TrackEvalException("Dataset-wide S has not been computed. Call compute_S_for_dataset() first.")
        
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.zeros((len(bboxes1), len(bboxes2)))

        if box_format == 'xywh':
            centers1 = np.stack([bboxes1[:, 0] + bboxes1[:, 2] / 2,
                                 bboxes1[:, 1] + bboxes1[:, 3] / 2], axis=1)
            centers2 = np.stack([bboxes2[:, 0] + bboxes2[:, 2] / 2,
                                 bboxes2[:, 1] + bboxes2[:, 3] / 2], axis=1)
        elif box_format == 'x0y0x1y1':
            centers1 = np.stack([(bboxes1[:, 0] + bboxes1[:, 2]) / 2,
                                 (bboxes1[:, 1] + bboxes1[:, 3]) / 2], axis=1)
            centers2 = np.stack([(bboxes2[:, 0] + bboxes2[:, 2]) / 2,
                                 (bboxes2[:, 1] + bboxes2[:, 3]) / 2], axis=1)
        else:
            raise TrackEvalException(f'Invalid box_format: {box_format}')

        dist_matrix = np.linalg.norm(centers1[:, np.newaxis, :] - centers2[np.newaxis, :, :], axis=2)
        dotd_scores = np.exp(-dist_matrix / S)  # Use S instead of fixed sigma

        return dotd_scores

    def _calculate_similarities_dotd(self, gt_dets_t, tracker_dets_t):
        return self._calculate_box_dot_distance(gt_dets_t, tracker_dets_t, box_format='xywh')
