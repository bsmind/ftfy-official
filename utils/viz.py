import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

import numpy as np

def merge_image(image_list, is_horizontal=True):
    if is_horizontal:
        return np.hstack(image_list)
    return np.vstack(image_list)

def get_rect(xy, w, h, color):
    return Rectangle(
        xy=xy, width=w, height=h,
        facecolor='none', edgecolor=color, linewidth=1
    )

# will be deleted
class RetrievalPlot_old(object):
    def __init__(self, n_examples, n_queries, top_k, colors,
                 image_size, patch_size,
                 n_scalars, n_lines, vmin=0, vmax=1):
        self.n_examples = n_examples
        self.n_queries = n_queries
        self.top_k = top_k
        self.colors = colors
        self.n_scalars = n_scalars
        self.n_lines = n_lines

        # for example images and bounding boxes
        fig1, ax1 = plt.subplots(1, n_examples)
        if n_examples == 1:
            ax1 = [ax1]

        imh = list()
        for ax in ax1:
            imh.append(ax.imshow(np.zeros(image_size), vmin=vmin, vmax=vmax))
            ax.axis('off')

        self.fig1 = fig1
        self.ax1 = ax1
        self.imh1 = imh

        # for top_k samples
        fig2, ax2 = plt.subplots(n_queries, n_examples)
        ax2 = ax2.ravel()
        imh = list()
        merged_size = (patch_size[0], patch_size[1]*(self.top_k+1))
        for ax in ax2:
            imh.append(ax.imshow(np.zeros(merged_size), vmin=vmin, vmax=vmax))
            ax.axis('off')

        self.fig2 = fig2
        self.ax2 = ax2
        self.imh2 = np.array(imh)

        fig3, ax3 = plt.subplots(1, n_scalars)
        if n_scalars == 1:
            ax3 = [ax3]
        else:
            ax3 = ax3.ravel()

        lineh = list()
        for i_ax, ax in enumerate(ax3):
            for _ in range(self.n_lines[i_ax]):
                line = ax.plot([], [])
                lineh.append(line[0])
            ax.set_autoscale_on(True)
            ax.autoscale_view(True, True, True)

        self.fig3 = fig3
        self.ax3 = ax3
        self.lineh = np.array(lineh)

        # show without blocking
        plt.ion()
        plt.show()
        plt.tight_layout()

    def _update_example(self, idx, image, ul_pos_xy, rect_sz):
        ax = self.ax1[idx]
        imh = self.imh1[idx]
        imh.set_data(image)

        # remove all patches and texts previouly drawn
        [p.remove() for p in reversed(ax.patches)]
        [p.remove() for p in reversed(ax.texts)]

        # draw bounding boxes
        for j, pos_xy in enumerate(ul_pos_xy):
            if j >= self.n_queries:
                break
            color = self.colors[j]
            for idx, xy in enumerate(pos_xy):
                ax.add_patch(
                    Rectangle(
                        xy=xy,
                        width=rect_sz[1], height=rect_sz[0],
                        edgecolor='none' if idx == 0 else color,
                        facecolor='none' if idx > 0 else color,
                        linewidth=1,
                        alpha=1.0 if idx > 0 else 0.5
                    )
                )
                if idx > 0:
                    ax.text(
                        x=xy[0],
                        y=xy[1],
                        s="{:d}".format(idx),
                        color=color
                    )

        #self.fig1.canvas.draw()
        #self.fig1.canvas.flush_events()

    def _update_top_k(self, idx, image, ul_pos_xy, rect_sz, qimg=None):
        ind = np.array([i for i in range(self.n_queries)], dtype=np.int32) * self.n_examples + idx
        ind = ind.tolist()
        ax = self.ax2[ind]
        imh = self.imh2[ind]

        h = rect_sz[0]
        w = rect_sz[1]
        for idx, pos_xy in enumerate(ul_pos_xy):
            if idx >= self.n_queries:
                break
            color = self.colors[idx]

            if qimg is None or len(qimg) == 0:
                img = merge_image([
                    image[xy[1]:xy[1]+h,xy[0]:xy[0]+w] for xy in pos_xy[:self.top_k+1]
                ])
            else:
                img = [qimg[idx]] + [
                    image[xy[1]:xy[1] + h, xy[0]:xy[0] + w] for xy in pos_xy[1:self.top_k + 1]
                ]
                img = merge_image(img)
            imh[idx].set_data(img)

            [p.remove() for p in reversed(ax[idx].patches)]
            ax[idx].add_patch(
                Rectangle(
                    xy=(0, 0),
                    width=w, height=h,
                    edgecolor=color,
                    facecolor='none',
                    linewidth=1
                )
            )

    def update(self, idx, image, ul_pos_xy_and_qimg, rect_sz):
        ul_pos_xy, qimg = ul_pos_xy_and_qimg
        qimg = np.squeeze(qimg)
        self._update_example(idx, image, ul_pos_xy, rect_sz)
        self._update_top_k(idx, image, ul_pos_xy, rect_sz, qimg)
        plt.draw()
        plt.pause(1)

    def update_scalar(self, value_list):
        for idx in range(self.n_scalars):
            data = value_list[idx]
            #offset = np.array(self.n_lines[:idx]).sum()
            for i_line in range(self.n_lines[idx]):
                h = self.lineh[i_line]
                new_data = np.append(h.get_ydata(), data[i_line])
                h.set_xdata(np.arange(len(new_data)))
                h.set_ydata(new_data)

            ax = self.ax3[idx]
            ax.relim()
            ax.autoscale_view(True, True, True)

        plt.draw()
        plt.pause(1)

    def hold(self):
        plt.ioff()
        plt.show()


# wil be deleted
class RetrievalPlot(object):
    def __init__(self, im_sz=(850, 1280), rect_sz=(204,204), k=5):
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        gs.update(left=0.03, right=0.99, bottom=0.11, top=0.99, wspace=0.01, hspace=0.03)

        ax1 = plt.subplot(gs[:2,:2])
        ax2 = plt.subplot(gs[0,2:])
        ax3 = plt.subplot(gs[1,2:])
        ax4 = plt.subplot(gs[2,:])

        self.main_im = ax1.imshow(
            np.zeros(im_sz, dtype=np.float32), vmin=0, vmax=1
        )
        ax1.axis('off')

        self.top_im = ax2.imshow(
            np.zeros((rect_sz[0]*2, rect_sz[1]*(k+1)), dtype=np.float32), vmin=0, vmax=1
        )
        ax2.axis('off')

        self.bot_im = ax3.imshow(
            np.zeros((rect_sz[0]*2, rect_sz[1]*(k+1)), dtype=np.float32), vmin=0, vmax=1
        )
        ax3.axis('off')

        self.fig = fig
        self.ax = [ax1, ax2, ax3, ax4]
        self.im_sz = im_sz
        self.rect_sz = rect_sz

        #plt.ion()
        plt.show()

    def update_images(self):
        pass

    def update_scalars(self):
        pass

    def hold(self):
        pass

class ScalarPlot(object):
    def __init__(self, n_lines, legends):
        fig, ax = plt.subplots(1, 1)
        lines = [ax.plot([],[], label=legends[i])[0] for i in range(n_lines)]
        ax.set_autoscale_on(True)
        ax.autoscale_view(True, True, True)

        self.fig = fig
        self.ax = ax
        self.lines = lines

        plt.legend(shadow=True, fancybox=True)
        plt.ion()
        plt.show()

    def update(self, values):
        if len(values) != len(self.lines):
            raise ValueError("values must be length of {:d}.".format(len(self.lines)))

        for line, v in zip(self.lines, values):
            data = np.append(line.get_ydata(), v)
            line.set_xdata(np.arange(len(data)))
            line.set_ydata(data)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)

        plt.draw()
        plt.pause(0.1)

    def hold(self):
        plt.ioff()
        plt.show()


