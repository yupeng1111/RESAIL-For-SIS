import importlib
import json
import multiprocessing
from multiprocessing import Process, Manager
import os
import pickle

import cv2.version
from imageio import imread
from imageio import imsave
import numpy as np
from PIL import Image
from pycocotools import mask as COCOmask
import torch
from tqdm import tqdm, trange

from data.utils import ADE20KPath
from data.utils import Root
from data.cityscapes_label import labels
from util.util import get_pth


# functions.
def multi_process_gather_list(iter_data, process_num):
    length = len(iter_data)
    each_length = length // process_num
    param_dict = [(each_length * i, each_length * i + each_length) for i in range(process_num)]
    if length % process_num != 0:
        param_dict.append((length - (length % each_length), length))

        process_num += 1

    process_list = [GatherList(*param, iter_data, thread_idx) for param, thread_idx in
                    zip(param_dict, range(process_num))]

    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    result = []
    for process in process_list:
        result.extend(process.gather)
    return result


def hash_rle(o):
    return hash(o['size'][0]) + hash(o['size'][1]) + hash(o['counts'])


def INT(*args):
    ret = []
    for a in args:
        ret.append(int(a))
    return tuple(ret)


def _segment_annotations(class_ids_and_segment_masks, data, image_path):
    image = imread(image_path)
    annotations = []
    segment_number_cnt = {}
    for class_id, segment_mask in class_ids_and_segment_masks:

        segment_mask_encoded_with_rle = COCOmask.encode(np.asfortranarray(segment_mask.astype(np.uint8)))
        segment_mask_area = segment_mask.sum()
        segment_mask_box = COCOmask.toBbox(segment_mask_encoded_with_rle)
        x, y, w, h = segment_mask_box
        h1 = int(y)
        h2 = int(y + h)
        w1 = int(x)
        w2 = int(x + w)
        if class_id in segment_number_cnt:
            segment_number_cnt[class_id] += 1
        else:
            segment_number_cnt[class_id] = 0
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)

        segment_rgb = image * np.expand_dims(segment_mask, axis=-1).astype(np.uint8)
        segment_rgb = segment_rgb[h1: h2, w1: w2]
        segment_rgb_file_name = os.path.basename(image_path).split(".")[0] + "_" + \
                                data.class_id2class_name[int(class_id)] + \
                                f"_classId{class_id}_" + \
                                str(segment_number_cnt[class_id]) + \
                                ".png"

        segment_rgb_path = os.path.join(data.instances_dir, segment_rgb_file_name)
        imsave(segment_rgb_path, segment_rgb)

        segment_annotation = {
            "image_path": image_path,
            "class_id": class_id,
            "segment_mask_area": segment_mask_area,
            "segment_mask_encoded_with_rle": segment_mask_encoded_with_rle,
            "class_name": data.class_id2class_name[int(class_id)],
            "bBox": segment_mask_box,
            'is_train': data.is_train,
            "segment_rgb_path": segment_rgb_path
        }

        annotations.append(segment_annotation)

    return annotations


# classes.

class Data:
    @staticmethod
    def find_data(data_name):
        dataset_lib = importlib.import_module('data.data_base')
        return dataset_lib.__dict__[data_name.lower().capitalize()]

    def __init__(self, opt, data_part, data_root, database_dir):
        self.image_paths = None

        self.extracted_root = None

    def __len__(self):
        return len(self.image_paths)

    @property
    def database_dir(self):
        return self.extracted_root


class Ade20k(Data):
    def __init__(self, data_part, opt, extracted_dir, preprocessor):
        super().__init__(opt, None, None, extracted_dir)

        path = ADE20KPath(data_part, opt.dataroot)
        path.set_path(self)

        # p_mode = opt.data_part
        # self.image_paths = dataset.image_paths
        # self.class_label_paths = dataset.class_label_paths
        # self.instance_label_paths = dataset.instance_label_paths

        self.instance_id2class_id = preprocessor.instance_id2class_id
        instance_info = os.path.join(opt.dataroot, 'objectInfo150.txt')
        id2name = np.loadtxt(instance_info, dtype=str, skiprows=1, usecols=[0, 4])
        self.class_id2class_name = {int(a[0]): a[1].replace(",", "") for a in id2name}

        self.instances_dir = os.path.join(extracted_dir, "ade20k", "instances", data_part)
        self.extracted_root = os.path.join(extracted_dir, "ade20k")

        self.is_train = 'training' == data_part
        if not os.path.isdir(self.instances_dir):
            os.makedirs(self.instances_dir)
        self.preprocessor = preprocessor

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        instance_label_path = self.instance_label_paths[index]

        instance_label = imread(instance_label_path)
        class_label = imread(self.class_label_paths[index])
        class_ids_and_segment_masks = self.preprocessor.parse_segments(
            instance_label,
            class_label,
        )

        annotations = _segment_annotations(class_ids_and_segment_masks, self, image_path)

        return annotations


# class ADE20kOutdoor(Data):
#
#     def __init__(self, data_part, dataset_root, database_dir, preprocessor):
#         super().__init__(data_part, None, database_dir)
#
#         root = Root(dataset_root)
#         self.image_paths = root.filter_pth(f'.*?/images/{data_part}/.*?jpg')
#         print(len(self.image_paths))
#
#         self.class_label_paths = root.filter_pth(f'.*?/annotations/{data_part}/.*?png')
#         self.instance_label_paths = root.filter_pth(f'.*?/annotations_instance/{data_part}/.*?png')
#
#         self.image_paths.sort(key=lambda string: string[-12: -4])
#         self.class_label_paths.sort(key=lambda string: string[-12: -4])
#         self.instance_label_paths.sort(key=lambda string: string[-12: -4])
#
#         for p1, p2, p3 in zip(self.instance_label_paths, self.image_paths, self.class_label_paths):
#             assert p1.replace('annotations_instance', '').replace('png', '') == \
#                 p2.replace('images', '').replace('jpg', ''), \
#                 '\n' + p1.replace('annotations_instance', '').replace('png', '') + \
#                 "\n" + p2.replace('images', '').replace('jpg', '')
#
#             assert os.path.basename(p1) == os.path.basename(p3)
#
#         # download from
#         # `https://github.com/CSAILVision/placeschallenge/blob/master/instancesegmentation/categoryMapping.txt`
#         # `https://github.com/CSAILVision/placeschallenge/blob/master/sceneparsing/objectInfo150.txt`
#         instance_info = os.path.join(dataset_root, 'objectInfo150.txt')
#         id2name = np.loadtxt(instance_info, dtype=str, skiprows=1, usecols=[0, 4])
#
#         category_mapping_path = os.path.join(dataset_root, "categoryMapping.txt")
#         instance_id2class_id = np.loadtxt(category_mapping_path, dtype=str, skiprows=1, usecols=[0, 1])
#
#         self.instance_id2class_id = {int(a[0]): int(a[1]) for a in instance_id2class_id}
#         self.class_id2class_name = {int(a[0]): a[1].replace(",", "") for a in id2name}
#
#         self.segments_dir = os.path.join(database_dir, "ade20k-outdoor", "segments", data_part)
#         self.database_root = os.path.join(database_dir, "ade20k-outdoor")
#         self.is_train = data_part == 'training'
#         if not os.path.isdir(self.segments_dir):
#             os.makedirs(self.segments_dir)
#         self.preprocessor = preprocessor
#
#     def __getitem__(self, index):
#         image_path = self.image_paths[index]
#         instance_label_path = self.instance_label_paths[index]
#
#         instance_label = imread(instance_label_path)
#         class_label = imread(self.class_label_paths[index])
#         class_ids_and_segment_masks = self.preprocessor.parse_segments(
#             instance_label,
#             class_label,
#         )
#
#         annotations = _segment_annotations(class_ids_and_segment_masks, self, image_path)
#
#         return annotations


class Cityscapes(Data):
    def __init__(self, data_part, data_root, database_dir, preprocessor):
        """
        Initialize cityscapes dataset, we search segment in each image.
        :param data_part: `val` for validation, 'train' for training.
        :param data_root: Cityscapes data root, where each image is resize as 256x512 resolution.
        :param database_dir: where the segment files are saved.
        """
        super().__init__(data_part, data_root, database_dir)
        root = Root(data_root)
        self.image_paths = root.filter_pth(f".*?/leftImg8bit/{data_part}/.*?_leftImg8bit.png")
        self.instance_label_paths = root.filter_pth(f".*?/gtFine/{data_part}/.*?_gtFine_instanceIds.png")

        self.image_paths.sort()
        self.instance_label_paths.sort()
        self.class_id2class_name = {label.id: label.name for label in labels}
        self.is_train = data_part == 'train'
        self.data_part = data_part
        self.database_root = os.path.join(database_dir, "cityscapes")
        self.segments_dir = os.path.join(database_dir, "cityscapes", "segments", self.data_part)
        if not os.path.isdir(self.segments_dir):
            os.makedirs(self.segments_dir)
        for p1, p2 in zip(self.image_paths, self.instance_label_paths):
            assert os.path.basename(p1.replace('leftImg8bit', '')) == os.path.basename(
                p2.replace('gtFine_instanceIds', ''))
        self.preprocessor = preprocessor

    def __getitem__(self, index):

        instance_label_path = self.instance_label_paths[index]
        image_path = self.image_paths[index]
        instance_label = imread(instance_label_path)
        class_ids_and_segment_masks = self.preprocessor.parse_segments(instance_label)

        annotations = _segment_annotations(class_ids_and_segment_masks, self, image_path)

        return annotations


class COCO(Data):
    def __init__(self, data_part, data_root, database_dir, preprocessor):

        super().__init__(data_part, data_root, database_dir)
        root = Root(data_root)
        # self.image_paths = root.filter_pth(f"/dataset/COCOinst/V1_2/COCO/images/{data_part}2017/(.*?).jpg")
        if data_part == 'training':
            data_part = 'train'
        elif data_part == 'validation':
            data_part = 'val'
        else:
            raise ValueError()
        self.image_paths = root.filter_pth(f"/data/zack/datasets/COCOinst/V1_2/COCO/images/{data_part}2017/(.*?).jpg")

        self.instance_label_paths = root.filter_pth(
            f"/data/zack/datasets/COCOinst/V1_2/COCO/inst_cls_annotations/{data_part}2017/(.*?).png")
        self.class_label_paths = root.filter_pth(
            f"/data/zack/datasets/COCOinst/V1_2/COCO/class_annotations/{data_part}2017/(.*?).png")

        self.image_paths.sort()
        self.instance_label_paths.sort()
        self.class_label_paths.sort()

        path = "/data/zack/datasets/COCOinst/V1_2/COCO/labels_coco.txt"
        self.class_id2class_name = {int(id[:-1]): name for id, name in np.loadtxt(path, dtype=str, usecols=[0, 1])}

        self.is_train = data_part == 'train'
        self.data_part = data_part
        self.database_root = os.path.join(database_dir, "coco")
        self.segments_dir = os.path.join(database_dir, "coco", "segments", self.data_part)
        if not os.path.isdir(self.segments_dir):
            os.makedirs(self.segments_dir)
        for p1, p2, p3 in zip(self.image_paths, self.instance_label_paths, self.class_label_paths):
            assert os.path.basename(p1.replace('jpg', '')) == os.path.basename(
                p2.replace('png', ''))
            assert os.path.basename(p2) == os.path.basename(
                p3)
        self.preprocessor = preprocessor

    def __getitem__(self, index):

        instance_label_path = self.instance_label_paths[index]
        image_path = self.image_paths[index]
        class_label_path = self.class_label_paths[index]

        instance_label = imread(instance_label_path)
        class_label = imread(class_label_path)

        class_ids_and_segment_masks = self.preprocessor.parse_segments(instance_label, class_label)

        annotations = _segment_annotations(class_ids_and_segment_masks, self, image_path)

        return annotations


class GatherList(Process):

    def __init__(self, index_from, index_to, iter_o, id):
        super().__init__()
        m = Manager()
        self.__gathered = m.list()
        self.__index_from = index_from
        self.__index_to = index_to
        self.__iter_o = iter_o
        self.__id = id

    def run(self) -> None:
        # Using tqdm to show the process. Outputs in screen maybe weird
        for index in tqdm(range(self.__index_from, self.__index_to), position=self.__id):
            self.__gathered.extend(self.__iter_o[index])

    @property
    def gather(self):
        return self.__gathered


class DataBase:
    @staticmethod
    def modify_commandline_options(parser):

        parser.add_argument('--extract_instance', action='store_true', help='Extracting instances in every image.')
        parser.add_argument('--extract_shape', action='store_true', help='Getting binary shape of every instance.')
        parser.add_argument('--cal_score', action="store_true")

        parser.add_argument('--extracted_file_root', type=str, default='./extracted')
        parser.add_argument('--calculating_gpu', type=int, default=0, help='The gpu used in processing dataset.')
        parser.add_argument('--score_balance', type=float, default=0.5)
        parser.add_argument('--topk', type=int, default=10)
        parser.add_argument('--extract_instance_threads', default=0, type=int,
                            help='How much threads are used during extracting instance.')

        return parser

    @staticmethod
    def annotation2pil(annotation):
        """
        Parsing an annotation to rgb image and mask of a segment.
        :param annotation: dict of annotations.
        :return: rgb(pil), binary mask(pil).
        """
        segment_rgb_path = annotation['segment_rgb_path']
        try:
            rgb = Image.open(segment_rgb_path)
        except:
            rgb = Image.fromarray(cv2.cvtColor(cv2.imread(segment_rgb_path), cv2.COLOR_BGR2RGB))
        mask = COCOmask.decode(annotation['segment_mask_encoded_with_rle'])
        x, y, w, h = annotation['bBox']
        x, y, w, h = INT(x, y, w, h)
        mask = mask[y: y + h, x: x + w]
        return rgb, Image.fromarray(mask.astype(np.uint8))

    def __init__(self, opt, preprocessor, dataset):
        # """
        # First, searching segments in each image and save them.
        # :param opt:
        # """

        self.opt = opt
        # self.segment_pool = ImagePool(opt.segment_pool_size)
        self.scores = {}
        self.data_name = opt.dataset_mode
        self.extracted_file_root = opt.extracted_file_root
        print(f'Preparing [{self.data_name}] segments database...')

        if self.opt.extract_instance:
            training_part = Data.find_data(self.data_name)('training', opt, extracted_dir=self.extracted_file_root,
                                                           preprocessor=preprocessor)
            validation_part = Data.find_data(self.data_name)('validation', opt, extracted_dir=self.extracted_file_root,
                                                             preprocessor=preprocessor)

            # if self.data_name == 'cityscapes':
            #
            #     train_part = Cityscapes(
            #         "train",
            #         data_root=opt.dataroot,
            #         database_dir=self.database_root,
            #         preprocessor=preprocessor
            #     )
            #     validation_part = Cityscapes(
            #         "val",
            #         data_root=opt.dataroot,
            #         database_dir=self.database_root,
            #         preprocessor=preprocessor
            #     )
            #
            # elif self.data_name == 'ade20k':
            #     train_part = Ade20k(
            #         'training',
            #         opt,
            #         extracted_dir=self.extracted_file_root,
            #         preprocessor=preprocessor,
            #     )
            #
            #     validation_part = Ade20k(
            #         'validation',
            #         opt,
            #         extracted_dir=self.extracted_file_root,
            #         preprocessor=preprocessor,
            #     )

            # elif self.data_name == 'ade20koutdoor':
            #     train_part = ADE20kOutdoor(
            #         "training",
            #         dataset_root=self.opt.data_root,
            #         database_dir=self.database_root,
            #         preprocessor=preprocessor
            #     )
            #     validation_part = ADE20kOutdoor(
            #         "validation",
            #         dataset_root=self.opt.data_root,
            #         database_dir=self.database_root,
            #         preprocessor=preprocessor
            #     )
            # elif self.data_name == 'coco':
            #     train_part = COCO(
            #         "training",
            #         data_root=self.opt.dataroot,
            #         database_dir=self.database_root,
            #         preprocessor=preprocessor
            #     )
            #     validation_part = COCO(
            #         "validation",
            #         data_root=self.opt.dataroot,
            #         database_dir=self.database_root,
            #         preprocessor=preprocessor
            #     )
            # else:
            #     raise NotImplementedError("No database instance implemented.")
            # self.train_part_anns = []
            # for annotation in tqdm(train_part, desc='searching'):
            #     self.train_part_anns.extend(annotation)

            if opt.extract_instance_threads == 0:
                threads_nums = multiprocessing.cpu_count()
            else:
                threads_nums = opt.extract_instance_threads

            self.train_annotations = multi_process_gather_list(training_part, threads_nums)

            # self.validation_part_anns = []
            # for annotation in tqdm(validation_part, desc='validation searching'):
            #     self.validation_part_anns.extend(annotation)

            self.validation_annotations = multi_process_gather_list(validation_part, threads_nums)

            # write annotations to disks.
            with open(
                    get_pth(self.extracted_file_root, self.data_name, 'annotations',
                            f'{self.data_name}_training_annotations.pkl'),
                    'wb'
            ) as f:
                pickle.dump(self.train_annotations, f)

            with open(
                    get_pth(self.extracted_file_root, self.data_name, 'annotations',
                            f'{self.data_name}_validation_annotations.pkl'),
                    'wb'
            ) as f:
                pickle.dump(self.validation_annotations, f)

        else:
            print(f"Reading annotations from [{get_pth(self.extracted_file_root, self.data_name, 'annotations')}]")
            with open(
                    get_pth(
                        self.extracted_file_root, self.data_name, 'annotations', f'{self.data_name}_training_annotations.pkl'
                    ),
                    'rb'
            ) as f:
                self.train_annotations = pickle.load(f)

            with open(
                    get_pth(
                        self.extracted_file_root,
                        self.data_name,
                        'annotations',
                        f'{self.data_name}_validation_annotations.pkl'
                    ),
                    'rb'
            ) as f:
                self.validation_annotations = pickle.load(f)

        self.annotations = self.train_annotations + self.validation_annotations

        self.each_class_number_training = {}  # store the number of segments of each class in training part.
        # if not self.data_name == 'coco':
        for annotation in self.train_annotations:
            class_name = annotation["class_name"]
            if class_name not in self.each_class_number_training.keys():
                self.each_class_number_training[class_name] = 1
            else:
                self.each_class_number_training[class_name] += 1

        for i, annotation in enumerate(self.annotations):
            annotation['id'] = i

        self.each_class = {}  # annotations in each class.
        for annotation in self.annotations:
            class_name = annotation['class_name']
            if class_name not in self.each_class.keys():
                self.each_class[class_name] = [annotation]
            else:
                self.each_class[class_name].append(annotation)

        if opt.extract_shape:
            for class_name in tqdm(self.each_class.keys(), position=1, desc='Processed classes'):
                shapes = []
                crops = []
                for annotation in tqdm(self.each_class[class_name], position=0, desc='class:' + class_name):
                    b = annotation['bBox']
                    x, y, w, h = b
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    shapes.append(w * h)
                    seg = COCOmask.decode(annotation['segment_mask_encoded_with_rle'])
                    seg_crop = seg[y: y + h, x: x + w]
                    bg = seg_crop

                    crop_t = torch.from_numpy(bg)
                    crop_t = crop_t.view(1, 1, *crop_t.shape)
                    # we unify the segment masks with the size of 128x128 to calculate in batches.
                    crop_t = torch.nn.functional.interpolate(crop_t.float(), size=(128, 128), mode='nearest')
                    crops.append(crop_t.squeeze())

                shapes_size = torch.Tensor(shapes)
                shapes_shapes = torch.stack(crops)

                p_size = get_pth(opt.extracted_file_root, self.data_name, 'tensors', class_name + '-size.pt')
                p_shape = get_pth(opt.extracted_file_root, self.data_name, 'tensors', class_name + '-shape.pt')
                torch.save(shapes_shapes, p_shape)
                torch.save(shapes_size, p_size)

        self.match_list = self.match_topk()
        self.rle2id = {hash_rle(ann['segment_mask_encoded_with_rle']): ann['id'] for ann in self.annotations}

    def search(self, shape, class_name=None, searched=True):
        if searched:
            try:
                return [self.annotations[idx] for idx in
                        self.match_list[self.rle2id[hash_rle(COCOmask.encode(np.asfortranarray(shape)))]]]
            except KeyError:
                return []
        else:
            raise NotImplementedError("used for demo.")

    def process_dataset(self):
        for class_name in tqdm(self.each_class.keys(), desc='calculate score...', position=1):
            p_shape_score = get_pth(
                self.extracted_file_root,
                self.data_name,
                "tensors",
                'scores',
                'shape_score-' + class_name + '.pt'
            )
            p_size_score = get_pth(
                self.extracted_file_root,
                self.data_name,
                "tensors",
                'scores',
                'size_score-' + class_name + '.pt'

            )
            if not (os.path.isfile(p_size_score) and os.path.isfile(p_shape_score)):

                shape_dst = torch.load(
                    get_pth(
                        self.extracted_file_root,
                        self.data_name,
                        'tensors',
                        class_name + '-shape.pt'
                    )
                ).cuda(self.opt.calculating_gpu)

                size_dst = torch.load(
                    get_pth(
                        self.extracted_file_root,
                        self.data_name,
                        'tensors',
                        class_name + '-size.pt'
                    )
                ).cuda(self.opt.calculating_gpu)
                size_scores = torch.mm(
                    size_dst.view(-1, 1),
                    torch.reciprocal(size_dst[:self.each_class_number_training[class_name]].view(1, -1))
                ).cpu()
                shape_scores = []
                for i in trange(
                        shape_dst.shape[0],
                        desc="class "
                             + class_name
                             + f" {self.each_class_number_training[class_name]} "
                               f"segments in TRAIN, all is {shape_dst.shape[0]}.",
                        position=0
                ):
                    shape = shape_dst[i]
                    shape_src = shape
                    shape_src = torch.repeat_interleave(shape_src.unsqueeze(0),
                                                        repeats=self.each_class_number_training[class_name],
                                                        dim=0)  # only match instances in TRAIN set.
                    sqrts = (shape_src - shape_dst[:self.each_class_number_training[class_name]]) ** 2

                    ssds = torch.sum(sqrts.view(self.each_class_number_training[class_name], -1), dim=1)
                    area1 = shape_src == 1
                    area1_sum = torch.sum(area1.cuda(self.opt.calculating_gpu).view(shape_src.shape[0], -1), dim=1,
                                          keepdim=True)
                    area2 = shape_dst[:self.each_class_number_training[class_name]] == 1
                    area2_sum = torch.sum(
                        area2.cuda(self.opt.calculating_gpu).view(self.each_class_number_training[class_name], -1),
                        dim=1, keepdim=True)
                    coff = torch.cat([area1_sum, area2_sum], dim=-1)
                    shape_score = ssds / torch.max(coff, dim=1)[0]
                    shape_scores.append(shape_score)

                shape_scores = torch.stack(shape_scores, dim=0).cpu()

                torch.save(shape_scores, p_shape_score)

                torch.save(size_scores, p_size_score)
            else:
                shape_scores = torch.load(p_shape_score)
                size_scores = torch.load(p_size_score)

            self.scores[class_name] = [shape_scores,
                                       torch.where(size_scores < 1, torch.reciprocal(size_scores), size_scores)]

    def match_topk(self):

        p = get_pth(
            self.opt.extracted_file_root,
            self.data_name,
            'match_list',
            f'top{self.opt.topk}-balance{self.opt.score_balance}.pkl'
        )
        if os.path.isfile(p) and (not self.opt.cal_score):
            with open(p, 'rb') as f:
                result = pickle.load(f)
        else:
            self.process_dataset()
            result = {}
            for class_name in tqdm(self.scores.keys(), desc="topk score"):
                annotations = self.each_class[class_name]
                scores = self.scores[class_name]
                size_score = torch.where(scores[1] < 2, torch.zeros_like(scores[1]), torch.ones_like(scores[1]))
                score = scores[0] + size_score
                try:
                    topk = torch.topk(score, dim=1, k=self.opt.topk + 1, largest=False)
                except RuntimeError as e:
                    topk = torch.topk(score, dim=1, k=1, largest=False)
                for i, ann in enumerate(annotations):
                    idx = ann['id']
                    if i < self.each_class_number_training[class_name]:
                        result[idx] = [annotations[j]['id'] for j in topk[1][i].tolist()[1:]]
                    else:
                        result[idx] = [annotations[j]['id'] for j in topk[1][i].tolist()]
                    # result[idx] = [annotations[j]['id'] for j in topk[1][i].tolist()]
            with open(p, 'wb') as f:
                pickle.dump(result, f)

        return result
