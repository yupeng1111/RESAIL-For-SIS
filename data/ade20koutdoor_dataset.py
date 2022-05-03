import os
import random
from glob import glob

from imageio import imread
import numpy as np
from PIL import Image

from data import data_base
from data.utils import Root
from data.base_dataset import BaseDataset, get_params, get_transform
from data.data_preprocessor import ADE20kOutdoorPreprocessor
from util.image import to_pil


class Ade20kOutdoorDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--data_root', type=str, default='./datasets/ADE20K_Outdoor')
        parser.add_argument('--ADEChallengeData2016', type=str, default='/dataset/ADE20K/V1_2/ADEChallengeData2016',
                            help='path to the directory that contains label images')
        parser.add_argument('--img_base_root', type=str, default='/dataset/ADE20K/V1_3/ADE20KInstance/images',  # no tail /
                            help='path to the directory that contains label images')
        parser.add_argument('--data_part', type=str, default='training', choices=["training", "validation"])
        parser.add_argument('--drop_out', type=float, default=[], nargs="+")
        parser = data_base.DataBase.modify_commandline_options(parser)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)

        return parser

    def initialize(self, opt):

        self.opt = opt
        root = Root(self.opt.data_root)
        self.image_paths = root.filter_pth(f'.*?/images/{opt.data_part}/.*?jpg')
        self.class_label_paths = root.filter_pth(f'.*?/annotations/{opt.data_part}/.*?png')
        self.instance_label_paths = root.filter_pth(f'.*?/annotations_instance/{opt.data_part}/.*?png')

        self.image_paths.sort(key=lambda string: string[-12: -4])
        self.class_label_paths.sort(key=lambda string: string[-12: -4])
        self.instance_label_paths.sort(key=lambda string: string[-12: -4])


        if not self.opt.use_cache:
            self.propocessor = ADE20kOutdoorPreprocessor(opt=opt)
            self.database = data_base.DataBase(opt, preprocessor=self.propocessor)
            self.propocessor.load_database(self.database)

        self.src_dir = opt.img_base_root
        self.dest_dir = os.path.join(opt.extracted_root, opt.dataset_mode, 'processed_images', f"drop{opt.drop_out}")

    def __len__(self):
        assert len(self.image_paths) == len(self.class_label_paths), \
            'The numbers of segmentations and images should be equal!'

        assert len(self.image_paths) == len(self.instance_label_paths), \
            'The numbers of segmentations and images should be equal!'

        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        class_label_path = self.class_label_paths[index]
        instance_label_path = self.instance_label_paths[index]

        image_pil = Image.open(image_path).convert('RGB')

        class_label_pil = Image.open(class_label_path)
        class_label_numpy = np.array(class_label_pil)

        instance_label_numpy = imread(instance_label_path)

        params = get_params(self.opt, class_label_pil.size)
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
            parameters = self.propocessor.preprocess_data(
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

        image = transform_image(to_pil(image_pil))
        class_label = transform_label(class_label_pil) * 255

        dic = {
            "image": image,
            "class_label": class_label,
            "image_path": image_path,
            "class_label_pth": class_label_path,
            "instance_label_pth": instance_label_path,
        }

        dic.update(t_dict)
        if self.opt.segloss:
            seg_label = transform_label(class_label_pil) * 255 - 1  # -1 ~ 149
            dic.update({"seg_label": seg_label})

        return dic
