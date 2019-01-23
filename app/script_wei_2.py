import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage.transform import downscale_local_mean, resize

def get_rect(x0, y0, w, h, edgecolor='red'):
    return Rectangle(
        xy=(x0, y0),
        width=w, height=h,
        facecolor='none', edgecolor=edgecolor, linewidth=1
    )

def get_db_img(im, down, x0, y0, rect_sz):
    patch = im[y0:y0+rect_sz[0], x0:x0+rect_sz[1]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch = downscale_local_mean(patch, (down, down, 1))
        patch = resize(patch, patch_size, order=1, mode='reflect', preserve_range=True)
    return patch


'''dict data
db: database
image: the original image used to build the database
filename: file name of the image in the local disk
other keys: query images and features obtained from different test conditions
'''
data = dict(np.load('../test_ms_fixed.npy').item())
patch_size = (208, 208)
top_k = 5
down_factors = [1, 2, 4, 8]

# play parameters
case_idx = 1 # choose a case
q_idx = 0

# query samples
q_examples = data.pop('q', None)
image = data.pop('image', None)
filename = data.pop('filename', 'Unknown')

# test_cases
cases = list(data.keys())
print('test cases: ', cases)

# show accuracy
for case in cases:
    acc = data[case]['acc']
    print('Accuracy @ {}: {}'.format(case, acc))


# example results
result = data.get(cases[case_idx])
down = result.get('down', None)
db = result.get('db', None)
acc = result.get('acc', None)
top_k_ind = result.get('top_k_ind', None)
top_k_ious = result.get('top_k_ious', None)

print("There are {:d} samples used to test under {:s}.".format(
    top_k_ind.shape[0], cases[case_idx]
))

# select one query example
q_feature = q_examples['features'][q_idx] # features of the queried image
q_x0 = q_examples['x0'][q_idx]            # x-coordinate of upper-left corner in the original image
q_y0 = q_examples['y0'][q_idx]            # y-coordinate of upper-left corner in the original image
q_img = q_examples['images'][q_idx]       # queried image with given condition

print(q_feature.shape, q_x0, q_y0, q_img.shape)

# visualize
fig, ax = plt.subplots(1, 2)
fig2, ax2 = plt.subplots(1, 5)

ax[0].imshow(np.squeeze(image), vmin=0, vmax=1) # show the original image
ax[0].axis('off')
ax[0].set_title('Overview')

ax[0].add_patch(get_rect(q_x0, q_y0, patch_size[1], patch_size[0], 'red')) # queried
for ii, idx in enumerate(top_k_ind[q_idx]):
    x0 = db['x0'][idx]
    y0 = db['y0'][idx]
    ax[0].add_patch(get_rect(x0, y0, patch_size[1], patch_size[0], 'yellow')) # top-k
    ax[0].text(x0, y0, 'top-{:d}'.format(ii+1))

    patch = get_db_img(image, down, x0, y0, patch_size)
    ax2[ii].imshow(np.squeeze(patch))
    ax2[ii].axis('off')
    ax2[ii].set_title('top-{:d}, {:.2f}'.format(
        ii+1, top_k_ious[q_idx][ii]
    ))

print('IoUs: {}'.format(top_k_ious[q_idx]))

ax[1].imshow(np.squeeze(q_img), vmin=0, vmax=1)
ax[1].axis('off')
ax[1].set_title('queried')

plt.show()










