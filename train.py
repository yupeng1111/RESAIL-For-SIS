"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
import os.path
from collections import OrderedDict
import sys
import time
import warnings
from functools import partial

from tqdm import tqdm

import data
from data.data_preprocessor import preprocess_batch
from options.train_options import TrainOptions
from test_fid import test_fid_client
from trainers.pix2pix_trainer import Pix2PixTrainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # parse options
    opt = TrainOptions().parse()
    # print options to help debugging
    print(' '.join(sys.argv))

    preprocess_batch_fn = partial(
        preprocess_batch, save_flag=False,
        src_dir=None,
        dest_dir=None
    )
    # load the dataset
    dataloader = data.create_dataloader(opt)

    # create trainer for our model
    trainer = Pix2PixTrainer(opt)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))

    # create tool for visualization
    visualizer = Visualizer(opt)
    all_data_t = 0
    all_train_t = 0
    t_data_1 = time.time()
    time_str = ""

    # Create fid tester.
    # used gpu#1 to do test work.
    if not opt.no_fid_tester:
        fid_calculator_client = test_fid_client.FidCalculatorClient(
            dataset_mode=opt.dataset_mode,
            name=opt.name,
            data_part="validation",
            batchSize="1",
            label_nc=opt.label_nc,
            FID="",
            gpu_ids="1",
            nThreads="20",
            no_instance="",
            data_root="/mnt/ShiYuPeng/Cityscapes",
            dataroot="/mnt/ShiYuPeng/Cityscapes",
            real_image_path=opt.real_image_dir,
            use_cache="",
            drop_out="0.0",
            port=2345,
        )

    for epoch in tqdm(iter_counter.training_epochs(), position=1):
        iter_counter.record_epoch_start(epoch)
        dataloader = tqdm(dataloader, position=0, desc=f"running epoch {epoch}")
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            data_i = preprocess_batch_fn(data_i)
            continue
            iter_counter.record_one_iteration()
            t_data_2 = time.time()
            t_data = t_data_2 - t_data_1
            all_data_t += t_data
            t_train_t1 = time.time()
            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            # train discriminator
            trainer.run_discriminator_one_step(data_i)
            t_train_t2 = time.time()
            t_train = t_train_t2 - t_train_t1
            all_train_t += t_train
            t_data_1 = time.time()
            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
                time_str = "data:%4fs;" % all_data_t + "train:%4fs" % all_train_t
                dataloader.set_postfix(time=time_str)
                all_data_t = 0
                all_train_t = 0

            if iter_counter.needs_displaying():
                visuals = OrderedDict([('input_label', data_i['class_label']),
                                       ('synthesized_image', trainer.get_latest_generated()),
                                       ('real_image', data_i['image'])])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                if not opt.no_fid_tester:
                    fid_calculator_client.cal_fid(epoch)
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

        dataloader.close()
        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
                epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

    if not opt.no_fid_tester:
        fid_calculator_client.close()
    print('Training was successfully finished.')


