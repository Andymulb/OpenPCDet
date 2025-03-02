import os
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    directory = "/home/avsjetsonagx1/diffSLAM/data/KITTI/2011_09_26_drive_0084/2011_09_26/2011_09_26_drive_0084_sync/velodyne_points/spatial_features"
    # Get sorted list of all .npy files in the directory
    files = sorted(f for f in os.listdir(directory) if f.endswith('.npy'))

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 8))
    im = None

    for file in files:
        sp_feat = np.load(os.path.join(directory, file))
        if im is None:
            im = ax.imshow(sp_feat, cmap='viridis')
            ax.set_title('Spatial Features')
            fig.colorbar(im, ax=ax)
        else:
            im.set_data(sp_feat)
            ax.draw_artist(ax.patch)
            ax.draw_artist(im)
        fig.canvas.flush_events()
        # plt.pause(0.01)
    
    plt.ioff()
    plt.close(fig)

if __name__ == '__main__':
    main()