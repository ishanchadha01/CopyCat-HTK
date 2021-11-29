import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from get_data import read_ark_file, read_htk_file, mlf_to_dict, model, get_feature_labels
from likelihoods import feature_max_ll
import os
import glob
import cv2
from tqdm import tqdm

def plot_mlf(mlf_file, fname, has_states=False):
    video_dict = mlf_to_dict(mlf_file)[fname]
    words = video_dict.keys()
    plt.subplot(2,1,1)
    for word in reversed(words):
        for state in range(len(video_dict[word])):
            height = video_dict[word][state][2]/1000 - video_dict[word][state][1]/1000
            bottom = video_dict[word][state][1]/1000
            plt.barh(word, height, left=bottom)
    return plt


def plot_feature(newMacros_file, mlf_file, dtype, data_dir, feature_id, fname):
    ''' Plots word boundaries as well as likelihoods
    '''  

    arr = []
    models = model(newMacros_file)
    video_dict = mlf_to_dict(mlf_file)[fname]
    words = video_dict.keys()
    if dtype == 'ark':
        frames = read_ark_file('{}/{}.ark'.format(data_dir, fname))
    else:
        frames = read_htk_file('./data/HList', '{}/{}.htk'.format(data_dir, fname))
    
    for word in words:
        gaussians = models[word][2]
        for frame in frames:
            ll = feature_max_ll(frame, gaussians, feature_id)
            arr.append(ll)
    plt.subplot(2,1,2)
    plt.plot(arr)
    return plt

    
def table_video(video_fp, viz_fp):
    cap = cv2.VideoCapture(video_fp)
    plot = cv2.imread(viz_fp)
    plot_x, plot_y, _ = plot.shape
    frames = []
    start_stop = []

    # Create start/stop lines
    for c in range(len(plot[0])):
        col = plot[:, c]
        black = np.count_nonzero(col==[0,0,0])
        if black > len(col):
            plot[:, c] = np.full((len(col), 3), [255,0,0])
            start_stop.append(c)
    start, stop = start_stop
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # if stop - start > total
    it = (stop - start) / total

    # Write frames and plot new frame and save to temp directory
    frame_num = 0
    prev = plot[:, start, :]
    with tqdm(total=total) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_x, frame_y, _ = frame.shape
            mult_x = int(np.ceil(frame_x / plot_x))
            mult_y = int(np.ceil(frame_y / plot_y))

            frame_num += 1
            pbar.update(1)
            plot[:, int(np.floor(start + it*(frame_num-1))), :] = prev
            prev = np.copy(plot[:, start + int(np.floor(it*frame_num)), :])
            plot[:, int(np.floor(start + it*frame_num)), :] = np.full((len(col), 3), [0,0,255])

            # Put video above plot
            w, h, _ = frame[::mult_y, ::mult_y, :].shape
            newFrame = np.zeros((plot_x + w, plot_y, 3))
            newFrame[:w, :h, :] = frame[::mult_y, ::mult_y, :]
            newFrame[w:, :, :] = plot
            cv2.imwrite('./temp/frame_{:04d}.png'.format(frame_num), newFrame)

    frame_size = (plot_x + w, plot_y)
    # out = cv2.VideoWriter('./plot.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 5, frame_size)
    # for filename in sorted(glob.glob(os.path.join('./temp', '*.png'))):
    #     img = cv2.imread(filename)
    #     out.write(img)
    # out.release()

    # Create video
    os.system('ffmpeg -r {} -f image2 -s {}x{} -i {}_%04d.png -codec:v libx264 -crf 25  -pix_fmt yuv420p {}'.format(5, frame_size[0], frame_size[1], './temp/frame', './data/video.mp4'))


if __name__=='__main__':
    plt.rcParams['axes.xmargin'] = 0
    results = './data/results'
    name = '01-31-21_Dhruva_4K+Depth.black_monkey_in_white_flowers.0000000010'
    for mlf in os.listdir(results):
        if mlf.endswith('mlf'):
            plt = plot_mlf(os.path.join(results, mlf), name)
            plt = plot_feature('./data/newMacros', os.path.join(results, mlf), 'ark', './data/ark',\
                0, name)
            plt.savefig('./data/{}_viz.png'.format(name))
    table_video('./data/{}.mkv'.format(name), './data/{}_viz.png'.format(name))