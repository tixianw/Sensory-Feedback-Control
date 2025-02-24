"""
Created on Tue Feb 6, 2024
@author: tixianw2
"""
import sys
sys.path.append("../") 
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
# from matplotlib.collections import LineCollection
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import gridspec 
import numpy as np
# from numpy.linalg import norm
# from scipy.signal import butter, filtfilt
np.seterr(divide='ignore', invalid='ignore')
# from tools import _diff, _aver, _diff_kernel, _aver_kernel
# import matplotlib.colors as mcolors
# import matplotlib as mpl
# from matplotlib.ticker import (MultipleLocator, 
                            #    FormatStrFormatter, 
                            #    AutoMinorLocator)
import os

def isninf(a):
    return np.all(np.isfinite(a))

folder = '../' # 'examples/'
folder_name = folder+'Data/'
file_name = 'bend_prop'

data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()

n_elem = data['model']['arm']['n_elem']
L = data['model']['arm']['L']
radius = data['model']['arm']['radius']
if radius[0]==radius[1]:
    r = ((2*radius) / (2*radius[0]))**2 * 50
else:
    r = ((2*radius) / (2*radius[0]))**2 * 100
E = data['model']['arm']['E']
final_time = data['model']['numerics']['final_time']
dt = data['model']['numerics']['step_size']
t = data['t']
s = data['model']['arm']['s']
s_mean = (s[1:] + s[:-1])/2
ds = s[1] - s[0]
position = data['position']
arm = data['arm']
muscle = data['muscle']
sensor = data['sensor']
save_step_skip = data['model']['numerics']['step_skip']
flags = data['model']['flags']
flag_target = flags[1]
flag_obs = flags[2]
if flag_target:
    target = data['model']['target']
if flag_obs:
    Obs = data['model']['obstacle']
    N_obs = Obs['N_obs']
    print(N_obs, 'obstacles')
    pos_obs = Obs['pos_obs']
    r_obs = Obs['r_obs']
    len_obs = Obs['len_obs']
if len(arm[-1]['orientation'][0,...])==3:
    orientation = arm[-1]['orientation'][:,1:,:-1,:]
elif len(arm[-1]['orientation'][0,...])==2:
    orientation = arm[-1]['orientation']
else:
    print('error!')
velocity = arm[-1]['velocity']
vel_mag = np.sqrt(np.einsum('ijn,ijn->in', velocity, velocity))

print('total_steps: ', len(t), ', final_time=', final_time)
print('target=', target[0,:])

base_radius = data['model']['arm']['base_radius']
A = np.pi * base_radius**2
I = A**2 / (4*np.pi)
EA = E * A
EI = E * I # (I[1:] + I[:-1]) / 2

video = 1 # 0
save_flag = 0

if video == 1:
    max_var = L*1.1
    min_var = -L/2
    idx = -1
    min_var_x = min(np.amin(position[idx][0,0,:])*1.1, np.amin(position[idx][-1,0,:])*1.1, min_var*1.01)
    max_var_x = max(np.amax(position[idx][-1,0,:])*1.1, np.amax(position[idx][0,0,:])*1.1, max_var*1.01)
    min_var_y = min(np.amin(position[idx][-1,1,:])*1.1, min_var*1.01)
    max_var_y = max_var*1.01
    # dist = sensor[idx]['dist'][:,:]
    fig = plt.figure(figsize=(10*0.6, 10*0.6))
    ax0 = fig.add_subplot(1, 1, 1)
    if save_flag:
        factor1 = 5 # min(int(1000 / save_step_skip), 1) # 5
        name = file_name
    else:
        factor1 = int(2000 / save_step_skip)
        name = 'trash'
    fps = 5 # 10
    try:
        os.mkdir(folder+'Videos/')
    except:
        pass
    video_name = folder+'Videos/' + name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        start = len(position)-1 # 0
        for k in range(start, len(position)):
            for jj in range(int((len(t)-1) / factor1)+1): # +1
                i = jj * factor1
                time = i / (len(t)-1) * final_time
                # idx = np.argmin(dist[i, :])
                ax0.cla()
                if flag_obs:
                    alpha0 = 0.8
                    name_obstacle = locals()
                    for o in range(N_obs):
                        name_obstacle['obstacle'+str(o)] = plt.Circle((pos_obs[o,0], pos_obs[o,1]), r_obs[o], color='grey', alpha=alpha0)
                        ax0.add_artist(name_obstacle['obstacle'+str(o)])
                ax0.scatter(position[k][i,0,:],position[k][i,1,:], s=r, marker='o', alpha=1,zorder=2)
                # ax0.scatter(position[k][i,0,idx],position[k][i,1,idx], s=r[idx], marker='o', color='red', alpha=1,zorder=3)
                ax0.text(L*0.1, max_var*1.05, 't: %.3f s'%(time), fontsize=12)
                ax0.scatter(target[i,0], target[i,1], s=200, marker='*', label='target point',zorder=1)
                # angle = np.linspace(0, 2*np.pi, 100)
                # distance = norm(target[0,:])
                # ax0.plot(target[0,0]+distance*np.cos(angle), target[0,1]+distance*np.sin(angle), ls='--', color='black')
                ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
                if not save_flag:
                    plt.pause(0.001)
                else:
                    writer.grab_frame()
            # break
            if not isninf(position[k]):
                break

plt.show()
