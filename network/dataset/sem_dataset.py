import os
import numpy as np

def filelist_by_zoomfactor(
        data_dir,
        output_dir = None,
        output_name = None
):
    import glob
    def txt2dict(fn):
        d = {}
        with open(fn) as f:
            for line in f:
                line = line.replace('$', '')
                tokens = line.split()

                if len(tokens) == 0: continue

                key = tokens[0]
                if len(tokens) == 1: d[key] = 'Unknown'
                elif len(tokens) > 1:
                    key = tokens[0]
                    value = " ".join(tokens[1:])
                    d[key] = value
        return d
    """For the XSEM data, it comes with .txt file containing zoom factor 
    information. In this function, .tif files are organized by the zoom 
    factor."""
    if not os.path.exists(data_dir):
        raise FileExistsError('Data path does not exist: ', data_dir)

    if output_dir is None:
        output_dir = data_dir

    glob_pathname = os.path.join(data_dir, '**', '*.tif')
    fileDict = {}

    for f in glob.glob(glob_pathname, recursive=True):
        zoom = 0
        txtname = os.path.splitext(f)[0] + '.txt'
        if os.path.exists(txtname):
            info = txt2dict(txtname)
            if 'CM_MAG' in info:
                zoom = int(info['CM_MAG'].split('.')[0])

        if zoom:
            fpath = f.replace(data_dir, "")
            if zoom in fileDict: fileDict[zoom].append(fpath)
            else: fileDict[zoom] = [fpath]
        else:
            print('Skip {:s}'.format(f))

    if output_name is not None:
        savepath = os.path.join(output_dir, output_name)
        np.save(savepath, fileDict)

    return fileDict

def split_train_test(files_by_zoom:dict, train_ratio=.7):
    """split train and test image set
    Files in each key (zoom) are distributed uniformly following round-robin
    fashion.
    """
    t_list = []
    tr_list = []
    add_to_tr = True
    for zoom, files in files_by_zoom.items():
        n_files = len(files)
        if n_files == 1:
            if add_to_tr: tr_list += files
            else: t_list += files
            add_to_tr = not add_to_tr
        else:
            n_train = np.int(np.round(n_files * train_ratio))
            np.random.shuffle(files)
            tr_list += files[:n_train]
            t_list += files[n_train:]

    return tr_list, t_list