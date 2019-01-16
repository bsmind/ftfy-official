'''
train dateset must include at least one example at a zoom factor
'''
import numpy as np

dataset_path = [
    '../data/sem_dataset_208_a.npy',
    '../data/sem_dataset_208_b.npy'
]

filenames = []
zooms = []
XYs = []
zoom_to_count = dict()

train_filenames = []
train_XYs = []
for path in dataset_path:
    dataset = dict(np.load(path).item())
    for f, xy_zoom in dataset.items():
        xy, z = xy_zoom
        zoom_to_count[z] = zoom_to_count.get(z, 0) + 1
        if zoom_to_count[z] == 1:
            train_filenames.append(f)
            train_XYs.append(xy)
        else:
            filenames.append(f)
            XYs.append(xy)

n_files = len(filenames) + len(train_filenames)
n_train = int(0.8 * n_files) - len(train_filenames)

ind = np.arange(len(filenames), dtype=np.int32)
np.random.shuffle(ind)

train_filenames += [filenames[i] for i in ind[:n_train]]
train_XYs += [XYs[i] for i in ind[:n_train]]

test_filenames = [filenames[i] for i in ind[n_train:]]
test_XYs = [XYs[i] for i in ind[n_train:]]

train_dataset = {'filenames': train_filenames, 'XY': train_XYs}
test_dataset = {'filenames': test_filenames, 'XY': test_XYs}

np.save("../data/train_sem_dataset_208.npy", train_dataset)
np.save("../data/test_sem_dataset_208.npy", test_dataset)

