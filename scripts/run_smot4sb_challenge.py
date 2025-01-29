#!/usr/bin/env python
import sys
import os
import argparse
import subprocess
from multiprocessing import freeze_support
import glob
import json

def convert_text_to_dict(file_pattern):
    """Convert a text file with space-separated values into a dictionary, allowing wildcards."""
    file_list = glob.glob(file_pattern)
    if not file_list:
        raise FileNotFoundError(f"No files match the pattern: {file_pattern}")
    
    file_path = file_list[0]  # Use the first matching file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    keys = lines[0].strip().split()
    values = list(map(float, lines[1].strip().split()))
    
    if len(keys) != len(values):
        raise ValueError("Mismatch between the number of keys and values in the file.")
    
    return dict(zip(keys, values))

def merge_results(existing_results, new_results):
    """Merge two dictionaries, giving priority to new_results for duplicate keys."""
    for key, new_value in new_results.items():
        if key in existing_results:
            old_value = existing_results[key]
            if old_value == new_value:
                print(f"Key '{key}' has the same value in both results: {old_value}")
            else:
                print(f"Key '{key}' has different values. Using MOT-Metrics value: {new_value}")
        existing_results[key] = new_value
    return existing_results

def set_configs(dataset_config, default_dataset_config, default_metrics_config):
    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False


    config = {**default_eval_config, **dataset_config, **default_metrics_config}  # Merge default configs

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    return eval_config, dataset_config, metrics_config

def run_evaluation(input_dir, output_dir, subset, use_metric_smot4sb, use_metric_mot):
    """Run the evaluation using TrackEval."""
    gt_folder = os.path.join(input_dir, 'ref')
    trackers_folder = os.path.join(input_dir, 'res')
    output_folder = output_dir

    if not os.path.isdir(gt_folder):
        raise Exception(f"Ground truth directory {gt_folder} doesn't exist.")
    if not os.path.isdir(trackers_folder):
        raise Exception(f"Trackers directory {trackers_folder} doesn't exist.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    freeze_support()

    # Update dataset_config with actual paths
    default_dataset_config = trackeval.datasets.SMOT4SBChallenge.get_default_dataset_config()
    dataset_config = {
        **default_dataset_config,
        'GT_FOLDER': gt_folder,
        'TRACKERS_FOLDER': trackers_folder,
        'OUTPUT_FOLDER': output_folder,
        'SPLIT_TO_EVAL': subset,
        # 'PLOT_CURVES': False,
    }

    all_results = {}
    
    if use_metric_smot4sb:
        # ===== SO-HOTA-based metrics =====
        default_metrics_config = {'METRICS': ['SO-HOTA'], 'THRESHOLD': 0.5}
        dataset_config["USE_SO_HOTA"] = True 
        eval_config, dataset_config, metrics_config = set_configs(dataset_config, default_dataset_config, default_metrics_config)
        
        # Run evaluation
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.SMOT4SBChallenge(dataset_config)]
        metrics_list = []
        metrics_list.append(trackeval.metrics.SO_HOTA(metrics_config))

        evaluator.evaluate(dataset_list, metrics_list)

        for _tracker in dataset_list[0].tracker_list:
            results = convert_text_to_dict(os.path.join(output_folder, _tracker, "SO-HOTA", "*_summary.txt"))
            all_results = merge_results(all_results, results)

    # ===== General metrics =====
    if use_metric_mot:
        default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
        dataset_config["USE_SO_HOTA"] = False
        eval_config, dataset_config, metrics_config = set_configs(dataset_config, default_dataset_config, default_metrics_config)
        
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.SMOT4SBChallenge(dataset_config)]
        metrics_list = []

        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric(metrics_config))
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')

        evaluator.evaluate(dataset_list, metrics_list)

        for _tracker in dataset_list[0].tracker_list:
            results = convert_text_to_dict(os.path.join(output_folder, _tracker, "MOT-Challenge-metrics", "*_summary.txt"))
            all_results = merge_results(all_results, results)

    # Save all results to scores.json
    scores_file_path = os.path.join(output_folder, "scores.json")
    with open(scores_file_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f'Evaluation completed. Results saved to {scores_file_path}.')

    # Make HTML file
    tracker = dataset_list[0].tracker_list[0]
    subprocess.run(["python", "generate_html.py", output_dir, tracker], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate tracking metrics using TrackEval.")
    parser.add_argument('input_dir', type=str, help="Path to the input directory containing 'ref' and 'res'.")
    parser.add_argument('output_dir', type=str, help="Path to the output directory for evaluation results.")
    parser.add_argument('subset', type=str, default="train", help="Subset to be evaluated")
    parser.add_argument('--metric-smot4sb', action='store_true', help="Use SO-HOTA metrics for SMOT4SB challenge.")
    parser.add_argument('--metric-mot', action='store_true', help="Use general MOT metrics (HOTA, CLEAR, Identity).")

    args = parser.parse_args()

    if not (args.metric_smot4sb or args.metric_mot):
        parser.error("At least one metric option (--metric-smot4sb or --metric-mot) must be specified.")

    # install_requirements()
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import trackeval

    run_evaluation(args.input_dir, args.output_dir, args.subset, args.metric_smot4sb, args.metric_mot)
