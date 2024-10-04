

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from env.vehicle import State


VISUALIZE = False
SAVE_LOG = False


def generate_color_gradient(color_len, gamma=0.8):
    """
    Generate an improved gamma-corrected color gradient with selected intermediate colors.

    Parameters:
        color_len (int): Length of the color sequence.
        gamma (float): Gamma value, default is 2.2.

    Returns:
        numpy.ndarray: Improved gamma-corrected color array with shape (color_len, 3).
    """
    # Generate a time variable from 0 to 1
    time = np.linspace(0, 1, color_len)

    # Choose a perceptually uniform color map (viridis)
    cmap = plt.get_cmap('winter')
    # inferno, 0.9, 0.2, 0.4, 0.5, 0.9
    # winter, 0.8, 0.2, 0.4, 0.5, 0.8

    # Extract colors from the color map at specific points
    start_color = cmap(0.2)[:3]  # Lighter color from the beginning of the colormap
    end_color = cmap(0.8)[:3]    # Darker color from the end of the colormap

    # # You can choose additional intermediate colors if needed
    mid_colors = np.array([
        cmap(0.45)[:3],  # Intermediate color
        cmap(0.55)[:3],  # Another intermediate color
    ])

    # Concatenate start_color, mid_colors, and end_color
    all_colors = np.concatenate([np.array([start_color]), mid_colors, np.array([end_color])], axis=0)

    # Interpolate colors based on time
    colors = np.array([np.interp(time, np.linspace(0, 1, len(c)), c) for c in all_colors.T]).T

    # Apply gamma correction to each color component
    gamma_corrected_colors = np.power(colors, 1/gamma)

    return gamma_corrected_colors

def draw_map(env_map, traj, save_path, draw_history=True):
    '''
    Params:
        env_map: Map object
        traj: list of state tuples
        save_path: str
    '''
    traj = [t.get_pos() for t in traj]
    final_traj = np.array(traj).T  # xs, ys, yaws
    # thick_traj = [t.get_pos() for t in env.vehicle.tmp_trajectory]
    # final_traj = (list([thick_traj[i][0] for i in range(len(thick_traj))]),
    #             list([thick_traj[i][1] for i in range(len(thick_traj))]), list([thick_traj[i][2] for i in range(len(thick_traj))]))

    TRAJ_RENDER_LEN = len(traj)
    TRAJ_COLORS = generate_color_gradient(TRAJ_RENDER_LEN)
    # start_color = (0.3,0.7,0.2)
    # target_color = 'gold'
    # path_color = 'chartreuse'
    start_color = 'aqua'
    target_color = 'greenyellow'
    path_color = 'gold'

    # plt.fill(*zip(*list(map.dest.create_box().coords)[:-1]), color='darkgreen',alpha=0.8)

    # ax.add_patch(plt.Polygon(xy=list(map.dest.create_box().coords)[:-1], color='b'))
    if hasattr(env_map, 'obstacles'):
        plt.axis('off')
        plt.xlim(env_map.xmin, env_map.xmax)
        plt.ylim(env_map.ymin, env_map.ymax)
        plt.gca().set_aspect('equal')
        for obs in env_map.obstacles:
            plt.fill(*zip(*list(obs.shape.coords)), color='gray')#,  alpha=0.9)
    else:
        ref_traj = env_map.data['ref_traj']
        xmin, xmax, ymin, ymax = np.min(ref_traj[:,0]), np.max(ref_traj[:,0]), np.min(ref_traj[:,1]), np.max(ref_traj[:,1])
        plt.axis('off')
        plt.xlim(xmin-5, xmax+5)
        plt.ylim(ymin-5, ymax+5)
        plt.gca().set_aspect('equal')
        plt.imshow(env_map.grid_map[::-1,:], cmap=ListedColormap(['white', 'gray']),\
                    extent=[env_map.xmin, env_map.xmax, env_map.ymin, env_map.ymax])
    # ax.add_patch(plt.Polygon(xy=list(map.start.create_box().coords)[:-1], color='g'))
    if len(traj) > 1:
        render_len = len(traj)
        # for i in range(len(self.vehicle.trajectory) - render_len):
        #     vehicle_box = self.vehicle.trajectory[i].create_box()
        #     pygame.draw.polygon(
        #         surface, TRAJ_COLORS[0], self._coord_transform(vehicle_box))
        # p
        for i in range(render_len):

            vehicle_box = State(traj[i]).create_box()
            if i == 0:
                continue
                plt.fill(*zip(*list(vehicle_box.coords)[:-1]), color='darkblue', edgecolor='aqua', alpha=1)
            elif i == render_len-1:
                if final_traj is not None and final_traj[0] is not None and final_traj[0][-1]!=traj[i][0]:
                    vehicle_box = State((final_traj[0][-1], final_traj[1][-1], final_traj[2][-1])).create_box()
                plt.fill(*zip(*list(vehicle_box.coords)[:-1]), color=TRAJ_COLORS[i], edgecolor=target_color, alpha=1, linewidth=1.5) # (0.3,0.7,0.2)
            else:
                if draw_history:
                    plt.fill(*zip(*list(vehicle_box.coords)[:-1]), color=TRAJ_COLORS[i], edgecolor='lightblue', alpha=0.6, linewidth=1.)
    if draw_history:
        vehicle_box = State(traj[0]).create_box()
        if final_traj is not None and final_traj[0] is not None and final_traj[0][0]!=traj[0][0]:
            vehicle_box = State((final_traj[0][0], final_traj[1][0], final_traj[2][0])).create_box()
            # old_box = State(traj[0]).create_box()
            # print('change box: ', list(vehicle_box.coords)[:-1], list(old_box.coords)[:-1])
        plt.fill(*zip(*list(vehicle_box.coords)[:-1]), color=TRAJ_COLORS[0], edgecolor=start_color, alpha=0.8, linewidth=1.5) # (1,0.2,0)

    if final_traj is not None and draw_history:
        xs, ys, yaws = final_traj
        if xs is not None and len(xs) > 1:
            # print('\n',len(xs))
            # print(traj[0], xs[0], xs[-1])
            # plt.plot(xs, ys, linewidth=2, color='r')
            plt.plot(xs, ys, linewidth=1.5, color=path_color)
            g_yaw = yaws[-1]# if yaw[-1]== yaw[-2] else yaws[-2]
            plt.arrow(xs[-1], ys[-1], 0.6*np.cos(g_yaw), 0.6*np.sin(g_yaw), width=.1, color=target_color, zorder=10)
            r_yaw = yaws[0] # if yaw[0]== yaw[1] else yaws[1]
            plt.arrow(xs[0], ys[0], 0.6*np.cos(r_yaw), 0.6*np.sin(r_yaw), width=.1, color=start_color, zorder=10)

    # draw the dest
    dest = env_map.dest.create_box()
    plt.fill(*zip(*list(dest.coords)[:-1]), edgecolor=target_color, alpha=0.8)

    def _get_img_array_from_fig(fig, dpi=180):
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img
    
    img_array = _get_img_array_from_fig(plt.gcf())

    if VISUALIZE:
        plt.show()
    if SAVE_LOG:
        plt.savefig(save_path, facecolor=plt.rcParams['axes.facecolor'])
    plt.close()

    return img_array