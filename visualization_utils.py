import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import imageio
import torch
import matplotlib.pyplot as plt
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "ffmpeg"

def rotate_camera_movie_volume(movie, filename=None, normalize=True, rotate=None, thresh=0.5, color='cool', fps=30, trans=False, distance=100):
    if normalize:
        movie -= movie.min()
        movie /= movie.max()

    if torch.is_tensor(movie):
        movie = movie.detach().cpu().numpy()

    pg.mkQApp()
    w = gl.GLViewWidget()
    if rotate:
        w.opts['distance'] = distance
        w.opts['elevation'] = 10
        w.opts['azimuth'] = 85
    else:
        w.opts['distance'] = distance
        w.opts['elevation'] = 10 # 50
        w.opts['azimuth'] = 85

    g = gl.GLGridItem()
    g.scale(10, 10, 1)
    w.addItem(g)

    [nFrames, _, _, _] = movie.shape

    if rotate:
        numAngles = 180 # 180
    else:
        numAngles = nFrames
    anglePerFrame = int(numAngles / nFrames)
    frame = -1

    writer = imageio.get_writer(filename, fps=fps)

    if color == 'cool':
        cmap = plt.get_cmap('cool')
    else:
        cmap = plt.get_cmap('Greys')

    for i in range(0, numAngles):
        if i % anglePerFrame == 0:
            frame += 1
            if frame == nFrames:
                frame = nFrames - 1

        data = np.squeeze(movie[frame, ...])

        data[data < thresh] = None
        rgba_data = 255*cmap(data)

        d2 = np.zeros(data.shape + (4,), dtype=np.ubyte)
        d2[..., 0] = rgba_data[..., 0]
        d2[..., 1] = rgba_data[..., 1]
        d2[..., 2] = rgba_data[..., 2]
        d2[..., 3] = rgba_data[..., 3]
        if trans:
           d2[..., 3] = 5

        v = gl.GLVolumeItem(d2, glOptions='translucent')
        v.translate(-40, -40, -40)
        w.addItem(v)

        ax = gl.GLAxisItem()
        w.addItem(ax)

        d = w.renderToArray((1008, 1008))
        # Convert to QImage
        image = pg.makeQImage(d)  # .save('custom3d_data/' + str(i) + '.png')
        # Convert to np array
        image = pg.imageToArray(image, copy=True, transpose=False)
        writer.append_data(image)
        w.removeItem(v)
        if rotate:
            if i < 90:
                j = .5
            else:
                j = -1
            w.orbit(2, j)
    writer.close()

def get_grid(slices, cols=5):
    [frames, h, w] = slices.shape

    if frames < cols:
        raise AssertionError('More columns than frames')

    if frames % cols != 0:
        raise AssertionError('Num frames must be divisible by num columns bc im lazy :(')

    rows = int(frames/cols)
    image_grid = np.zeros((rows * h, cols * w))
    count = 0

    for row in range(0, rows):
        for col in range(0, cols):
            image_grid[row*h:(row+1)*h, col*w:(col+1)*w] = slices[count, ...]
            count = count + 1

    return image_grid

def get_slices(movie, dim=None, axis=2):
    [frames, h, w, d] = movie.shape
    if dim is None:
        dim = d // 2

    slices = np.zeros((frames, h, w))

    for i in range(0, frames):
        if axis == 2:
            slices[i, ...] = movie[i, :, :, dim]
        if axis == 1:
            slices[i, ...] = movie[i, :, dim, :]
        if axis == 0:
            slices[i, ...] = movie[i, dim, :, :]
    return slices