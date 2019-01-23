import matplotlib.pyplot as plt
import numpy as np
from network.dataset.create_dataset import ImageContainer, InteractiveImageHandler

def load_dataset(filename='sem_dataset_208.npy'):
    dataset = np.load(filename).item()
    return dataset

def load_dataset_files(filename):
    dataset = load_dataset(filename)
    files = list(dataset.keys())
    return files

if __name__ == '__main__':
    import network.dataset.sem_dataset as sem

    data_dir = '/home/sungsooha/Desktop/Data/ftfy/data_hxnsem_selected'
    dataset_name = '../data/sem_dataset_208.npy'

    # dict: zoom => files
    files_by_zoom = sem.filelist_by_zoomfactor(data_dir)
    exclude_list = load_dataset_files('../data/sem_dataset_208_a.npy') + \
        load_dataset_files('../data/sem_dataset_208_b.npy')# filelist to exclude
    file_list = [] # all files



    # dict: file => zoom
    file_to_zoom = dict()
    for zoom, filenames in files_by_zoom.items():
        for f in filenames:
            if f in exclude_list: continue
            file_to_zoom[f] = zoom
            file_list.append(f)

    images = ImageContainer(data_dir, file_list)

    img_size = (850, 1280)
    patch_size = (208, 208)
    img = np.zeros(img_size, dtype=np.float32)
    key_actions = {
        ',': lambda: images.prev(),
        '.': lambda: images.next(),
        'o': lambda key, info: images.save_info(key, info)
    }

    fig = plt.figure()
    ax = fig.add_subplot(111)

    h = ax.imshow(img, vmin=0, vmax=1)
    ih = InteractiveImageHandler(h, image_size=img_size, patch_size=patch_size)
    ih.connect()
    ih.register_key_actions(key_actions)
    plt.show()

    #images.save_to_file(file_list, dataset_name, file_to_zoom)