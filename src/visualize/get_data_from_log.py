# This script get and save data of network input/output from the log file of real vehicle test.
# Steps:
# 1. get the gridmap data by running convert_gridmap_info.py in CyberRock_IVFC/src/parking/planning/rl_planning/
# 2. copy the grid_map_info.pkl under CyberRock_IVFC/src/parking/planning/rl_planning/data/log to /HOPE/src/log/test
# 3. rename the grid_map_info.pkl (eg.parallel.pkl) as the same in this script
# 4. run this script (sometimes exchange the dest and start first), the data will be saved in /HOPE/data/grid_map
# 5. move other scenario files into /HOPE/data/grid_map/hidden, only keep the target scenario file
# 6. run the visualization script in /HOPE/src/visualize/vis-knn.py (use the gridmap-based env)

import os
import pickle as pkl
import numpy as np


test_file = './log/test/vertical.pkl'


if __name__ == '__main__':
    with open(test_file, 'rb') as f:
        info = pkl.load(f)

    # 
    # grid_map_info = {
    #     'grid_map': grid_map,
    #     'oridin_roi': [env_map.xmin, env_map.xmax, env_map.ymin, env_map.ymax],
    #     'start': env_map.start.get_pos(),
    #     'dest': env_map.dest.get_pos(),
    #     'thick_traj': np.array(thick_traj),
    #     'vehicle_traj': np.array([t.get_pos() for t in vehicle_traj]),
    # }

    grid_map = info['grid_map']
    origin_roi = info['oridin_roi']
    start = info['start']
    dest = info['dest']
    thick_traj = info['thick_traj']
    vehicle_traj = info['vehicle_traj']
    xy_resolution = 0.1

    # start, dest = dest, start

    data_pkl = {
        'gridmap': grid_map,
        'map_range': origin_roi,
        'starts': [start],
        'goal': dest,
        'ref_traj_thick': thick_traj,
        'ref_traj': vehicle_traj,
        'xy_resolution': xy_resolution
    }

    test_file_name = os.path.basename(test_file)
    pkl_file = os.path.join('../data/grid_map', test_file_name)
    with open(pkl_file, 'wb') as f:
        pkl.dump(data_pkl, f)

