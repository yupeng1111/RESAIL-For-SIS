"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
from collections import OrderedDict
from functools import partial

import data
from data.data_preprocessor import preprocess_batch
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.image import compose_retrieval_and_modified_images
from util.visualizer import Visualizer
from util import html
from glob import glob
from util import image

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
if opt.FID:
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '%s_%s_final' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))
# from util.image import grid_img_result, saveim
# imgs = []
# for i, data_i in enumerate(dataloader):
#     data_i = preprocess_batch(data_i, False, None, None)
#     imgs.append(data_i['retrieval_image'])
# for i, data_i in enumerate(dataloader):
#     data_i = preprocess_batch(data_i, False, None, None)
#     imgs.append(data_i['modified_image'])
# for i, data_i in enumerate(dataloader):
#     data_i = preprocess_batch(data_i, False, None, None)
#     imgs.append(data_i['image'])

# im = grid_img_result(imgs, cols=1, line=0)
# saveim(im, 'temp.png')
# assert False
# test
for i, data_i in enumerate(dataloader):
    data_i = preprocess_batch(data_i, False, None, None)

    generated = model(data_i, mode='inference')
    img_path = data_i['image_path']

    if opt.FID:
        for b in range(generated.shape[0]):

            print('process image... %s' % img_path[b])
            visuals = OrderedDict(
                [
                    ('synthesized_image', generated[b]),
                    ('retrieval_image', data_i['retrieval_image'][b]),
                ]
            )
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        continue

    if i * opt.batchSize >= opt.how_many:
        break


    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict(
            [
                ('input_label', data_i['class_label'][b]),
                ('modified_image', data_i['modified_image'][b]),
                ('real_image', data_i['image'][b]),
                ('retrieval_image', data_i['retrieval_image'][b]),
                ('synthesized_image', generated[b]),
            ]
        )

        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

if opt.FID:
    exit(0)


webpage.save()

im_Ps = []
dirs = glob(os.path.join(web_dir, 'images', '*/'))
for dir in dirs:

    im_Ps.extend(sorted(glob(os.path.join(dir, '*'))))
cols = opt.how_many
result_img = image.grid_img_result(im_Ps, cols=cols)
image.saveim(result_img, os.path.join(web_dir, 'images', 'result.png'))
print(f"Results were saved in `{os.path.join(web_dir, 'images', 'result.png')}`.")
