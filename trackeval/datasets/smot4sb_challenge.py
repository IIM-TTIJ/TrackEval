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
    """Dataset class for SMOT4SB Challenge"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['bird'],  # Valid: ['pedestrian']
            'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'val', 'pub-test', 'private-test'
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
        # 1) First, just initialize the base class (_BaseDataset)
        _BaseDataset.__init__(self)

        # 2) Merge config
        if config is None:
            config = {}
        self.config = utils.init_config(
            user_config=config,
            default_config=self.get_default_dataset_config(),
            dataset_name=self.get_name()
        )

        self.data_is_zipped = self.config['INPUT_AS_ZIP']
        self.do_preproc = self.config['DO_PREPROC']

        # 3) Configure main directory and subfolders
        #  (do not use MOTChallenge specific 'BENCHMARK' or 'SKIP_SPLIT_FOL')
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], self.config['SPLIT_TO_EVAL'])
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], self.config['SPLIT_TO_EVAL'])
        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol
        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        # 4) Class information (bird only)
        self.valid_classes = ['bird']
        self.class_name_to_class_id = {'bird': 1}
        self.class_list = [
            cls.lower() if cls.lower() in self.valid_classes else None
            for cls in self.config['CLASSES_TO_EVAL']
        ]
        if not all(self.class_list):
            raise TrackEvalException('Invalid class specified. Only "bird" is valid.')
        # Valid class ID
        self.valid_class_numbers = [1]

        # 5) Get sequence information (seqmaps or SEQ_INFO)
        self.seq_list, self.seq_lengths = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences selected to be evaluated.')

        # 6) Check for existence of GT files
        if not self.config['INPUT_AS_ZIP']:
            for seq in self.seq_list:
                gt_path = self.config['GT_LOC_FORMAT'].format(gt_folder=self.gt_fol, seq=seq)
                if not os.path.isfile(gt_path):
                    raise TrackEvalException(f'GT file not found for sequence: {seq}')
        else:
            # In case of ZIP
            zip_path = os.path.join(self.gt_fol, 'data.zip')
            if not os.path.isfile(zip_path):
                raise TrackEvalException(f'GT data.zip not found: {zip_path}')

        # 7) Tracker list and file existence check
        if self.config['TRACKERS_TO_EVAL'] is None:
            if os.path.isdir(self.tracker_fol):
                self.tracker_list = os.listdir(self.tracker_fol)
            else:
                self.tracker_list = []
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        else:
            if len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list):
                self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
            else:
                raise TrackEvalException('Mismatch between TRACKERS_TO_EVAL and TRACKER_DISPLAY_NAMES.')

        if self.config['INPUT_AS_ZIP']:
            # ZIP check
            for tracker in self.tracker_list:
                tracker_zip = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
                if not os.path.isfile(tracker_zip):
                    raise TrackEvalException(f'Tracker zip not found: {tracker_zip}')
        else:
            # TXT check
            for tracker in self.tracker_list:
                for seq in self.seq_list:
                    tracker_path = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                    if not os.path.isfile(tracker_path):
                        raise TrackEvalException(f'Tracker file not found: {tracker_path}')

        # 8) Configuration for SO-HOTA
        self.use_so_hota = self.config['USE_SO_HOTA']
        if self.use_so_hota:
            self.S = None  # Dataset-wide normalization factor for DotD
            self.compute_S_for_dataset()  # Caliculate normalization factor S for DotD
            if self.output_sub_fol == None or self.output_sub_fol == "":
                self.output_sub_fol = "SO-HOTA"
            else:
                self.output_sub_fol += "_SO-HOTA"
        else:
            if self.output_sub_fol == None or self.output_sub_fol == "":
                self.output_sub_fol = "MOT-Challenge-metrics"
            else:
                self.output_sub_fol += "_MOT-Challenge-metrics"
        print("[DEBUG]", self.output_sub_fol)

    def _get_seq_info(self):
        seq_list = []
        seq_lengths = {}
        if self.config["SEQ_INFO"]:
            seq_list = list(self.config["SEQ_INFO"].keys())
            seq_lengths = self.config["SEQ_INFO"]

            # If sequence length is 'None' tries to read sequence length from .ini files.
            for seq, seq_length in seq_lengths.items():
                if seq_length is None:
                    ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                    if not os.path.isfile(ini_file):
                        raise TrackEvalException('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])

        else:
            if self.config["SEQMAP_FILE"]:
                # 
                seqmap_file = self.config["SEQMAP_FILE"]
            else:
                # If there is a direct designation, use it.
                if self.config["SEQMAP_FOLDER"] is None:
                    seqmap_file = os.path.join(self.config['GT_FOLDER'], 'seqmaps', self.config['SPLIT_TO_EVAL'] + '.txt')
                else:
                    seqmap_file = os.path.join(self.config["SEQMAP_FOLDER"], self.config['SPLIT_TO_EVAL'] + '.txt')
            if not os.path.isfile(seqmap_file):
                print('no seqmap found: ' + seqmap_file)
                raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
            with open(seqmap_file) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0 or row[0] == '':
                        continue
                    seq = row[0]
                    seq_list.append(seq)
                    ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                    if not os.path.isfile(ini_file):
                        raise TrackEvalException('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
        return seq_list, seq_lengths

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
        """ Preprocess data for a single sequence for a single class ("bird") ready for evaluation.
        Inputs:
            - raw_data: dict containing the data for the sequence already read in by get_raw_seq_data().
            - cls: the class to be evaluated (should be "bird").
        Outputs:
            - data: dict containing all of the information that metrics need to perform evaluation.
                Fields:
                [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets]: integers.
                [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            For this dataset, we assume only "bird" class is valid (class_id=1).
            Any other class IDs in GT or tracker data will raise an error (if do_preproc=True).
            
            The following preprocessing steps are performed:
                1) Remove GT dets that are zero_marked or not equal to the target class.
                2) Optionally remove tracker dets that do not match valid GT (if do_preproc=True).
                3) Relabel IDs so they are contiguous from 0..N-1.
                4) Ensure ID uniqueness per timestep.
        """

        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        # The class id for "bird" (or the class given by cls)
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets',
                    'tracker_confidences', 'similarity_scores']
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

            # If tracker outputs any class other than our valid class (cls_id),
            # raise an error or handle it (we assume only "bird" is valid).
            if len(tracker_classes) > 0 and not np.all(tracker_classes == cls_id):
                raise TrackEvalException(
                    f"Evaluation is only valid for class ID={cls_id}. "
                    f"Found non-bird class in seq={raw_data['seq']} at t={t} "
                    f"(tracker_classes={tracker_classes})."
                )

            # Do we do additional preproc? If so, remove tracker dets matched with invalid GT
            to_remove_tracker = np.array([], np.int64)
            if self.do_preproc and gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                # Check for invalid GT classes (anything not in self.valid_class_numbers)
                invalid_classes = np.setdiff1d(np.unique(gt_classes), self.valid_class_numbers)
                if len(invalid_classes) > 0:
                    # Show which invalid class IDs are found
                    msg_invalid = ' '.join(str(x) for x in invalid_classes)
                    raise TrackEvalException(
                        "Attempting to evaluate using invalid gt classes. "
                        "Please either check your gt data or disable preprocessing. "
                        f"Invalid classes found in seq={raw_data['seq']} t={t}: {msg_invalid}"
                    )

                # Perform Hungarian matching (on similarity scores) to remove any tracker
                # dets matched to invalid GT or to be removed. 
                # [Note] Since we have no "distractor" concept for bird-only dataset,
                # we won't do a "distractor" removal. 
                # However, if we wanted to remove GT with zero_marked=0 or non-bird class,
                # we could do a similar check here. 
                matching_scores = similarity_scores.copy()
                # Threshold small similarities to 0
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > np.finfo('float').eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                # (Optional) If you needed to remove certain GT => you could define mask here
                # But for a pure "bird" dataset with no distractors, we skip that.

                # Example: If you have any GT that you want to treat as "to remove," you could do:
                # is_to_remove = ...
                # to_remove_tracker = match_cols[is_to_remove]

                # Since we don't have explicit distractors, we can leave "to_remove_tracker" empty
                # unless you have other criteria.

            # Apply preprocessing to remove undesired tracker dets
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Remove gt detections that are zero_marked or not in the target class
            gt_to_keep_mask = (gt_zero_marked != 0) & (gt_classes == cls_id)

            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

    # Re-label IDs so there are no empty IDs
    if len(unique_gt_ids) > 0:
        unique_gt_ids = np.unique(unique_gt_ids)
        gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
        gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
        for t in range(raw_data['num_timesteps']):
            if len(data['gt_ids'][t]) > 0:
                data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int64)
    if len(unique_tracker_ids) > 0:
        unique_tracker_ids = np.unique(unique_tracker_ids)
        tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
        tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
        for t in range(raw_data['num_timesteps']):
            if len(data['tracker_ids'][t]) > 0:
                data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int64)

    # Record overview statistics
    data['num_tracker_dets'] = num_tracker_dets
    data['num_gt_dets'] = num_gt_dets
    data['num_tracker_ids'] = len(unique_tracker_ids)
    data['num_gt_ids'] = len(unique_gt_ids)
    data['num_timesteps'] = raw_data['num_timesteps']
    data['seq'] = raw_data['seq']

    # Ensure again that ids are unique per timestep after preproc
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
        
    def _calculate_box_dot_distance(self, bboxes1, bboxes2, box_format='xywh'):
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
        dotd_scores = np.exp(-dist_matrix / self.S)  # Use S instead of fixed sigma

        return dotd_scores

    def _calculate_similarities_dotd(self, gt_dets_t, tracker_dets_t):
        return self._calculate_box_dot_distance(gt_dets_t, tracker_dets_t, box_format='xywh')
