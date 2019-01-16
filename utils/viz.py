import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def merge_image(image_list, is_horizontal=True):
    if is_horizontal:
        return np.hstack(image_list)
    return np.vstack(image_list)

class RetrievalPlot(object):
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
