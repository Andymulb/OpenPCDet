import argparse
import glob
from pathlib import Path

import open3d
from visual_utils import open3d_vis_utils as V

import numpy as np
import torch
import time

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import cProfile
import pstats
from io import StringIO

import matplotlib.pyplot as plt
import os


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--max_num_frames', type=int, default=None, help='specify the max number of frames to visualize')
    parser.add_argument('--visualization', action='store_true', default=False, help='whether to visualize the results')
    parser.add_argument('--spatial_features_output_dir', type=str, default=None, help='specify the output directory for spatial features')
    parser.add_argument('--max_spatial_features', type=int, default=3000, help='maximal number of spatial features to export')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    if args.visualization:
        plt.ion()
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        im = ax[0, 0].imshow(np.zeros((100, 100)), cmap='viridis')
        ax[0, 0].set_title('Pseudo-Image (Summed Across Channels)')
        fig.colorbar(im, ax=ax[0, 0])

        im_fine = ax[0, 1].imshow(np.zeros((100, 100)), cmap='viridis')
        ax[0, 1].set_title('Fine-Grained Spatial Features')
        fig.colorbar(im_fine, ax=ax[0, 1])

        im_medium = ax[1, 0].imshow(np.zeros((100, 100)), cmap='viridis')
        ax[1, 0].set_title('Medium-Scale Spatial Features')
        fig.colorbar(im_medium, ax=ax[1, 0])

        im_large = ax[1, 1].imshow(np.zeros((100, 100)), cmap='viridis')
        ax[1, 1].set_title('Large-Scale Spatial Features')
        fig.colorbar(im_large, ax=ax[1, 1])

        plt.show(block=False)


    # Create output directory
    if args.spatial_features_output_dir:
        os.makedirs(args.spatial_features_output_dir, exist_ok=True)
        
    with torch.no_grad():
        if args.visualization:
            vis = open3d.visualization.Visualizer()
            vis.create_window()

        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, batch_dict, _ = model.forward(data_dict)
            
            # Extract spatial features
            pseudo_image = batch_dict['spatial_features'].squeeze().cpu().numpy()

            multi_scale_features = batch_dict['spatial_features_2d'].squeeze().cpu().numpy()
            for i in range(3):
                if i == 0:
                    fine_grained_spatial_features = multi_scale_features[0]
                    # fine_grained_spatial_features = fine_grained_spatial_features.sum(axis=0)
                elif i == 1:
                    medium_scale_spatial_features = multi_scale_features[128]
                    # medium_scale_spatial_features = medium_scale_spatial_features.sum(axis=0)
                else:
                    large_scale_spatial_features = multi_scale_features[256]
                    # large_scale_spatial_features = large_scale_spatial_features.sum(axis=0)
            
            if args.spatial_features_output_dir:
                if idx < args.max_spatial_features:
                    # Save as .npy file
                    out_file = os.path.join(args.spatial_features_output_dir, f"spatial_features_{idx:05d}.npy")
                    np.save(out_file, pseudo_image)
                    
                    # # Save as PNG file
                    # png_file = os.path.join(args.spatial_features_output_dir, f"spatial_features_{idx:05d}.png")
                    # plt.imsave(png_file, pseudo_image, cmap='viridis')
            
            if args.visualization:
                im.set_data(pseudo_image.sum(axis=0))
                ax[0, 0].draw_artist(ax[0, 0].patch)
                ax[0, 0].draw_artist(im)
                fig.canvas.flush_events()

                im_fine.set_data(fine_grained_spatial_features)
                ax[0, 1].draw_artist(ax[0, 1].patch)
                ax[0, 1].draw_artist(im_fine)
                fig.canvas.flush_events()

                im_medium.set_data(medium_scale_spatial_features)
                ax[1, 0].draw_artist(ax[1, 0].patch)
                ax[1, 0].draw_artist(im_medium)
                fig.canvas.flush_events()

                im_large.set_data(large_scale_spatial_features)
                ax[1, 1].draw_artist(ax[1, 1].patch)
                ax[1, 1].draw_artist(im_large)
                fig.canvas.flush_events()


            # Count how many boxes were predicted for each class
            labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            unique_labels, counts = np.unique(labels, return_counts=True)
            logger.info("Object counts per class:")
            for label, count in zip(unique_labels, counts):
                logger.info(f"  Class {label}: {count} detected")

            if args.visualization:
                V.draw_scenes(
                    vis, points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )
                time.sleep(0.1)

            if args.max_num_frames and idx >= args.max_num_frames - 1:
                break

        if args.visualization:
            vis.destroy_window()
            plt.ioff()
            plt.close(fig)

    logger.info('Demo done.')


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    s = StringIO()
    # sort profiling results by cumulative time and show the top 50 results
    sortby = 'time' # 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(50)

    print(s.getvalue())
