import os
import random

from imageio import imread
from PIL import Image

from data import data_base
from data.utils import Root
from data.base_dataset import BaseDataset, get_params, get_transform
from data.data_preprocessor import CityscapesPreprocessor


class CityscapesDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # todo resize the Cityscapes dataset
        parser.add_argument('--data_root', type=str, default='./datasets/Cityscapes',
                            help='path to the directory that contains label images')
        parser.add_argument('--data_part', type=str, default='training', choices=["training", "validation"])
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=35)
        parser.set_defaults(aspect_ratio=2.0)
        parser.set_defaults(batchSize=16)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')

        parser = data_base.DataBase.modify_commandline_options(parser)
        parser.add_argument('--drop_out', type=float, default=[], nargs="+")
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.data_part = opt.data_part
        super().initialize(opt)

        part = 'train' if "train" in self.data_part else 'val'

        root = Root(self.opt.data_root)

        self.image_paths = root.filter_pth(f".*?/leftImg8bit/{part}/.*?_leftImg8bit.png")
        self.instance_label_paths = root.filter_pth(f".*?/gtFine/{part}/.*?_gtFine_instanceIds.png")
        self.label_paths = root.filter_pth(f".*?/gtFine/{part}/.*?_gtFine_labelIds.png")

        # names = ["frankfurt_000001_035864_leftImg8bit.png",
        #             "frankfurt_000001_041074_leftImg8bit.png",
        #             "frankfurt_000001_042384_leftImg8bit.png",
        #             "frankfurt_000001_048196_leftImg8bit.png",
        #             "lindau_000000_000019_leftImg8bit.png",
        #             "lindau_000013_000019_leftImg8bit.png",]
        # names = ['munster_000170_000019_leftImg8bit.png']
        # self.image_paths = [p for p in self.image_paths if os.path.basename(p) in names]
        # self.instance_label_paths = [ p for p in self.instance_label_paths if os.path.basename(p).replace("gtFine_instanceIds", "leftImg8bit") in names]
        # self.label_paths = [p for p in self.label_paths if os.path.basename(p).replace("gtFine_labelIds", "leftImg8bit") in names]

        self.image_paths.sort()
        self.instance_label_paths.sort()
        self.label_paths.sort()

        self.preprocessor = CityscapesPreprocessor(opt=opt, preserved_ratio=0.9)
        self.database = data_base.DataBase(opt, preprocessor=self.preprocessor)
        self.preprocessor.load_database(self.database)

    def __len__(self):
        assert len(self.image_paths) == len(self.instance_label_paths), \
            'The numbers of segmentations and images should be equal!'
        for p1, p2 in zip(self.image_paths, self.instance_label_paths):
            assert os.path.basename(p1.replace('leftImg8bit', '')) == os.path.basename(
                p2.replace('gtFine_instanceIds', ''))

        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        instance_label_path = self.instance_label_paths[index]
        label_path = self.label_paths[index]

        image_pil = Image.open(image_path).convert('RGB')
        instance_label_numpy = imread(instance_label_path)
        label = Image.open(label_path)

        params = get_params(self.opt, label.size)
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

        parameters = self.preprocessor.preprocess_data(
            instance_label_numpy,
            image_pil,
            transform_image,
            transform_label,
            drop_out=drop_out_value
        )

        image_pil = transform_image(image_pil)
        class_label = transform_label(label) * 255

        return_dict = {
            "image": image_pil,
            "class_label": class_label,
            "image_path": self.image_paths[index],
            "class_label_pth": self.label_paths[index],
            "instance_label_pth": self.instance_label_paths[index],
            "parameters": parameters
        }
        return return_dict
