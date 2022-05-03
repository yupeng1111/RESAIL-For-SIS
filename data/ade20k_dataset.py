import os
import random
from glob import glob

from imageio import imread
import numpy as np
from PIL import Image

from data import data_base
from data.utils import ADE20KPath
from data.base_dataset import BaseDataset, get_params, get_transform
from data.data_preprocessor import ADE20kPreprocessor, warp_points
from util.image import to_pil

# classes.



class Ade20kDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--data_root", type=str, default='./datasets/ADE20K/')

        # parser.add_argument('--outdoor', action='store_true')
        # todo if test ade20k-outdoor from ade20k.
        # parser.add_argument('--ade20koutdoor_root', type=str,default='./datasets/ADE20K_Outdoor')

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
        # if opt.outdoor:
        #     outdoor_root = Root(self.opt.ade20koutdoor_root)
        #     paths = outdoor_root.filter_pth(".*?/annotations/.*?/.*?png")
        #     self.names = names = [os.path.basename(p) for p in paths]

        self.data_part = opt.data_part

        path = ADE20KPath(self.data_part, opt.dataroot)
        path.set_path(self)

        # root = Root(opt.dataroot)

        # self.image_paths = path.image_paths
        # self.class_label_paths = path.class_label_paths
        # self.instance_label_paths = path.instance_label_paths

        # if opt.outdoor:
        #     self.image_paths = [p for p in self.image_paths if os.path.basename(p).replace('jpg', 'png') in names]
        #     self.class_label_paths = [p for p in self.class_label_paths if os.path.basename(p) in self.names]
        #     self.instance_label_paths = [p for p in self.instance_label_paths if os.path.basename(p) in self.names]

        # names = ['1953']
        # t1 = self.image_paths[0]
        # t2 = self.class_label_paths[0]
        # t3 = self.instance_label_paths[0]
        # self.image_paths = [os.path.join(os.path.dirname(t1), os.path.basename(t1)[:12] + n + os.path.basename(t1)[-4:]) for n in names]
        # self.class_label_paths = [os.path.join(os.path.dirname(t2), os.path.basename(t2)[:12] + n + os.path.basename(t2)[-4:]) for n in names]
        # self.instance_label_paths = [os.path.join(os.path.dirname(t3), os.path.basename(t3)[:12] + n + os.path.basename(t3)[-4:]) for n in names]

        # self.image_paths.sort(key=lambda string: string[-12: -4])
        # self.class_label_paths.sort(key=lambda string: string[-12: -4])
        # self.instance_label_paths.sort(key=lambda string: string[-12: -4])
        # self.check_path()

        # for p1, p2, p3 in zip(self.image_paths, self.class_label_paths, self.instance_label_paths):
        #     n1 = os.path.basename(p1)
        #     n2 = os.path.basename(p2)
        #     n3 = os.path.basename(p3)
        #     assert n1.split('.')[0] == n2.split('.')[0]
        #     assert n2.split('.')[0] == n3.split('.')[0]
        # print(len(self.image_paths))
        # print('check over')
        # assert False

        self.propocessor = ADE20kPreprocessor(opt=opt, preserved_ratio=0.9)
        self.database = data_base.DataBase(opt, preprocessor=self.propocessor, dataset=self)
        self.propocessor.load_database(self.database)

    # def check_path(self):
    #     for p1, p2, p3 in zip(self.instance_label_paths, self.image_paths, self.class_label_paths):
    #         assert p1.replace('annotations_instance', '').replace('png', '') == \
    #             p2.replace('images', '').replace('jpg', ''), \
    #             '\n' + p1.replace('annotations_instance', '').replace('png', '') + \
    #             "\n" + p2.replace('images', '').replace('jpg', '')
    #
    #         assert os.path.basename(p1) == os.path.basename(p3), f'{p1} does not match {p3}'

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
