import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_rect(x0, y0, w, h, edgecolor='red'):
    return Rectangle(
        xy=(x0, y0),
        width=w, height=h,
        facecolor='none', edgecolor=edgecolor, linewidth=1
    )

'''dict data
db: database
image: the original image used to build the database
filename: file name of the image in the local disk
other keys: query images and features obtained from different test conditions
'''
data = dict(np.load('test.npy').item())
patch_size = (208, 208)
top_k = 5

# database
'''dict db 
features: features
x0, y0: upper-left corners of image patches in the original image
image: (None) image patches if available 
'''
db = data.pop('db', None)
image = data.pop('image', None)
filename = data.pop('filename', 'Unknown')

# test_cases
cases = list(data.keys())
case_idx = 0

# example results
'''dict result
q: (dict) queried information
top_k_ind: pre-computed top-k indices
top_k_ious: pre-computed top_k IoUs 
'''
result = data.get(cases[case_idx])
q_examples = result.get('q', None)
top_k_ind = result.get('top_k_ind', None)
top_k_ious = result.get('top_k_ious', None)

print("There are {:d} samples used to test under {:s}.".format(
    top_k_ind.shape[0], cases[case_idx]
))

# select one query example
q_idx = 0
q_feature = q_examples['features'][q_idx] # features of the queried image
q_x0 = q_examples['x0'][q_idx]            # x-coordinate of upper-left corner in the original image
q_y0 = q_examples['y0'][q_idx]            # y-coordinate of upper-left corner in the original image
q_img = q_examples['images'][q_idx]       # queried image with given condition

print(q_feature.shape, q_x0, q_y0, q_img.shape)

# visualize
fig, ax = plt.subplots(1, 2)

ax[0].imshow(np.squeeze(image), vmin=0, vmax=1) # show the original image
ax[0].axis('off')

ax[0].add_patch(get_rect(q_x0, q_y0, patch_size[1], patch_size[0], 'red')) # queried
for idx in top_k_ind[q_idx]:
    x0 = db['x0'][idx]
    y0 = db['y0'][idx]
    ax[0].add_patch(get_rect(x0, y0, patch_size[1], patch_size[0], 'yellow')) # top-k
print('IoUs: {}'.format(top_k_ious[q_idx]))

ax[1].imshow(np.squeeze(q_img), vmin=0, vmax=1)
ax[1].axis('off')

plt.show()










