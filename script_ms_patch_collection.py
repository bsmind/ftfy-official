import utils.data as DATA


if __name__ == '__main__':
    base_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
    data_dir = 'scene'#''campus'

    task_1 = False # create multi-scaled patch sets
    task_2 = False # generate random triplet examples out of the patch sets
    task_3 = False # generate random matched examples out of the patch sets
    task_4 = True # generate random image retrieval examples out of the patch sets

    psz_low = 13 # patch size at the lowest resolution (scale)
    psz_final = 128
    down_factors = [1, 2, 4, 6, 8, 10]
    iou_range = [(0.7, 1.0), (0.5, 0.7), (0.3, 0.5)]
    n_max_corners = 50

    n_triplet_samples = 1000000
    n_matched_pairs   =   50000
    n_query_per_group = 2

    if task_1:
        print('Create multi-scaled patch sets: %s' % data_dir)
        DATA.create_ms_patchset(
            base_dir=base_dir,
            data_dir=data_dir,
            psz_low=psz_low,
            down_factors=down_factors,
            iou_range=iou_range,
            n_max_corners=n_max_corners,
            psz_final=psz_final
        )

    if task_2:
        print('Generate triplet examples: %s' % data_dir)
        DATA.generate_triplet_samples(
            base_dir=base_dir,
            data_dir=data_dir + '_patch',
            n_samples=n_triplet_samples,
            debug=True
        )

    if task_3:
        print('Generate matched pairs: %s' % data_dir)
        DATA.generate_matched_pairs(
            base_dir=base_dir,
            data_dir=data_dir + '_patch',
            n_samples=n_matched_pairs,
            debug=True
        )

    if task_4:
        print('Generate retrieval test set: %s' % data_dir)
        DATA.generate_image_retrieval_samples(
            base_dir=base_dir,
            data_dir=data_dir + '_patch',
            n_query_per_group=n_query_per_group,
            debug=True
        )
