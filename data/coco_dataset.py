"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
import random

from PIL import Image
from imageio import imread

from data import BaseDataset, data_base
from data.base_dataset import get_params, get_transform
from data.data_preprocessor import COCOPreprocessor
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from data.utils import Root
from util.image import to_pil


class CocoDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--coco_no_portraits', action='store_true')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=182)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        parser.add_argument('--data_part', type=str, default='training', choices=["training", "validation"])
        parser.add_argument('--instance_label_dir', type=str, default='/dataset/COCOinst/V1_2/COCO/inst_cls_annotations',
                            help='path to the directory that contains label images')
        parser.add_argument("--image_dir", type=str, default="/dataset/COCOinst/V1_2/COCO/images")
        parser.add_argument("--class_label_dir", type=str, default="/dataset/COCOinst/V1_2/COCO/class_annotations")
        parser.add_argument("--coco_data_root", type=str, default="/dataset/COCOinst/V1_2/COCO")
        parser.add_argument('--drop_out', type=float, default=[], nargs="+")
        parser.add_argument("--img_base_root", type=str, default="/dataset/COCOinst/V1_2/COCO/images")
        parser = data_base.DataBase.modify_commandline_options(parser)
        return parser


    def initialize(self, opt):
        self.opt = opt

        if opt.data_part == "validation":
            self.data_part = "val"
        else:
            self.data_part = "train"

        class_label_root = Root(self.opt.class_label_dir)
        instance_label_root = Root(self.opt.instance_label_dir)
        image_root = Root(self.opt.image_dir)

        self.image_path_list = image_root.filter_pth(
            f".*?/COCOinst/V1_2/COCO/images/{self.data_part}2017/(.*?).jpg"
        )
        self.instance_label_path_list = instance_label_root.filter_pth(
            f".*?/COCOinst/V1_2/COCO/inst_cls_annotations/{self.data_part}2017/(.*?).png"
        )
        self.class_label_path_list = class_label_root.filter_pth(
            f".*?/COCOinst/V1_2/COCO/class_annotations/{self.data_part}2017/(.*?).png"
        )

        self.image_path_list.sort(key=lambda p: os.path.basename(p)[:-4])
        self.instance_label_path_list.sort(key=lambda p: os.path.basename(p)[:-4])
        self.class_label_path_list.sort(key=lambda p: os.path.basename(p)[:-4])

        if not opt.use_cache:
            self.preprocessor = COCOPreprocessor(opt=opt, preserved_ratio=0.9)
            self.database = data_base.DataBase(opt, preprocessor=self.preprocessor)
            self.preprocessor.load_database(self.database)

        self.src_dir = opt.img_base_root
        self.dest_dir = os.path.join(opt.extracted_root, opt.dataset_mode, 'processed_images', f"drop{opt.drop_out}")

    def __len__(self):
        assert len(self.image_path_list) == len(self.class_label_path_list) and \
               len(self.class_label_path_list) == len(self.instance_label_path_list)
        for p1, p2, p3 in zip(self.image_path_list, self.instance_label_path_list, self.class_label_path_list):
            n1 = int(os.path.basename(p1)[:-4])
            n2 = int(os.path.basename(p2)[:-4])
            n3 = int(os.path.basename(p3)[:-4])
            assert n1 == n2 and n2 == n3, print(n1, n2, n3)
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        class_label_path = self.class_label_path_list[index]
        instance_label_path = self.instance_label_path_list[index]

        image_pil = Image.open(image_path).convert('RGB')
        instance_label_numpy = imread(instance_label_path)
        class_label_numpy = imread(class_label_path)
        label = Image.open(class_label_path)

        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_image = get_transform(self.opt, params)

        if self.opt.use_cache:
            modified_image_path = image_path.replace(self.src_dir, os.path.join(self.dest_dir, "modified")).replace('jpg', 'png')
            retrieval_image_path = image_path.replace(self.src_dir, os.path.join(self.dest_dir, "retrieval")).replace('jpg', 'png')
            modified_image_pil = Image.open(modified_image_path).convert('RGB')
            retrieval_image_pil = Image.open(retrieval_image_path).convert('RGB')
            modified_image = transform_image(to_pil(modified_image_pil))
            retrieval_image = transform_image(to_pil(retrieval_image_pil))
            t_dict = {
                "modified_image": modified_image,
                "retrieval_image": retrieval_image
            }
        else:
            if len(self.opt.drop_out) == 0:
                drop_out_value = 0
            elif len(self.opt.drop_out) == 1:
                drop_out_value = self.opt.drop_out[0]
            elif len(self.opt.drop_out) == 2:
                drop_out_value = random.uniform(self.opt.drop_out[0], self.opt.drop_out[1])
            else:
                raise ValueError(f"drop_out_value length > 2")

            parameters = self.preprocessor.preprocess_data(
                instance_label_numpy,
                class_label_numpy,
                image_pil,
                transform_image,
                transform_label,
                drop_out=drop_out_value
            )
            t_dict = {
                "parameters": parameters
            }
        image_pil = transform_image(image_pil)
        class_label = transform_label(label) * 255
        class_label[class_label == 255] = self.opt.label_nc

        return_dict = {
            "image": image_pil,
            "class_label": class_label,
            "image_path": self.image_path_list[index],
            "class_label_pth": self.class_label_path_list[index],
            "instance_label_pth": self.instance_label_path_list[index],
        }

        return_dict.update(t_dict)

        return return_dict

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else opt.phase

        label_dir = os.path.join(root, '%s_label' % phase)
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        if not opt.coco_no_portraits and opt.isTrain:
            label_portrait_dir = os.path.join(root, '%s_label_portrait' % phase)
            if os.path.isdir(label_portrait_dir):
                label_portrait_paths = make_dataset(label_portrait_dir, recursive=False, read_cache=True)
                label_paths += label_portrait_paths

        image_dir = os.path.join(root, '%s_img' % phase)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if not opt.coco_no_portraits and opt.isTrain:
            image_portrait_dir = os.path.join(root, '%s_img_portrait' % phase)
            if os.path.isdir(image_portrait_dir):
                image_portrait_paths = make_dataset(image_portrait_dir, recursive=False, read_cache=True)
                image_paths += image_portrait_paths

        if not opt.no_instance:
            instance_dir = os.path.join(root, '%s_inst' % phase)
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)

            if not opt.coco_no_portraits and opt.isTrain:
                instance_portrait_dir = os.path.join(root, '%s_inst_portrait' % phase)
                if os.path.isdir(instance_portrait_dir):
                    instance_portrait_paths = make_dataset(instance_portrait_dir, recursive=False, read_cache=True)
                    instance_paths += instance_portrait_paths

        else:
            instance_paths = []

        return label_paths, image_paths, instance_paths
