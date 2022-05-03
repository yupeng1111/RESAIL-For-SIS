# import os
#
# import data
# from data.data_preprocessor import preprocess_batch
#
#
# def ade_cache():
#     class Opt:
#         def __init__(self):
#             pass
#
#     opt = Opt()
#     opt.data_part = "training"
#     opt.ADE20KInstance = '/dataset/ADE20K/V1_3/ADE20KInstance/'
#     opt.ADEChallenghData2016 =
#     dataloader = data.create_dataloader(opt)
#     for index, batch_data in enumerate(dataloader):
#         print(f"caching [{index} / {len(dataloader)}].")
#         preprocess_batch(
#             batch_data,
#             True,
#             src_dir=args.img_base_root,
#             dest_dir=os.path.join(opt.database_root, opt.dataset_mode, 'processed_images', f"drop{opt.drop_out}")
#         )
#     exit(0)
#
#
