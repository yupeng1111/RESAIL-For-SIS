"""
Watermark of these code.
@Author : Yupeng Shi
@Email : csypshi@gmail.com
@File : data_preprocesser.py
@Project : ${PROJECT_NAME}
Please reserve this statement.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import copy
import json
import math
import os
import random
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

import settings
from util.image import to_pil, region_color_transfer, to_box, compose_retrieval_and_modified_images


# classes.
class BasePreprocessor:
    def __init__(self, preserved_ratio, opt):
        self.preserved_ratio = preserved_ratio
        self.opt = opt

    def load_database(self, database):
        setattr(self, "database", database)

    def parse_segments(self, *args, **kwargs):
        raise NotImplementedError("func. `parse_segment` should be overwrite.")

    def _process_data(self, *args, image_pil=None, fg_mask_id_list=None, transform_image, transform_label, **kwargs):
        assert hasattr(self, 'database'), '`database` is not loaded.'
        shape_w, shape_h = image_pil.size
        segment_class_ids_and_masks = self.parse_segments(*args, **kwargs)
        modified_segment_list = []  # numpy data.
        segment_mask_list = []
        is_fgs_list = []
        retrieval_segment_mask_list = []
        retrieval_segment_rgb_list = []

        for segment_class_id, segment_mask in segment_class_ids_and_masks:

            src_segment = np.expand_dims(segment_mask, axis=-1) * np.array(image_pil)
            matched_annotations = self.database.search(segment_mask, searched=True)
            if len(matched_annotations) == 0:
                continue

            h1, h2, w1, w2 = to_box(segment_mask, bounding_box=False)
            annotation = matched_annotations[0]

            pil_segment_rgb, pil_segment_mask = self.database.annotation2pil(annotation)

            # change resolution and color
            # color
            annotation_idx = np.random.randint(self.opt.topk)
            # if self.opt.isTrain == False: annotation_idx = 0
            annotation_used_for_color_transfer = matched_annotations[annotation_idx]
            rgb, mask = self.database.annotation2pil(annotation_used_for_color_transfer)

            color_transferred_segment = region_color_transfer(
                src_segment,
                segment_mask,
                np.array(rgb),
                np.array(mask)
            )

            # resolution
            color_transferred_segment = to_pil(color_transferred_segment)
            ratio = np.random.uniform(0.5, 1)
            size = color_transferred_segment.size
            new_size = int(size[0] * ratio), int(size[1] * ratio)
            color_transferred_segment = color_transferred_segment.resize(
                new_size,
                Image.BICUBIC
            )
            color_transferred_segment = color_transferred_segment.resize(
                size,
                Image.BICUBIC
            )

            segment_mask_list.append(segment_mask)

            modified_segment_list.append(color_transferred_segment)
            is_fgs_list.append(segment_class_id in fg_mask_id_list)

            retrieval_segment_mask = np.zeros(shape=(shape_h, shape_w), dtype=np.uint8)  # h, w, c
            retrieval_segment_mask[h1: h2, w1: w2] = np.array(
                pil_segment_mask.resize((w2 - w1, h2 - h1), Image.NEAREST))

            retrieval_segment_rgb = np.zeros(shape=(shape_h, shape_w, 3), dtype=np.uint8)  # h, w, c
            retrieval_segment_rgb[h1: h2, w1: w2] = np.array(
                pil_segment_rgb.resize((w2 - w1, h2 - h1), Image.BICUBIC))

            retrieval_segment_mask_list.append(retrieval_segment_mask)
            retrieval_segment_rgb_list.append(retrieval_segment_rgb)

        # There should be none images without any segment.
        if not (len(segment_class_ids_and_masks) == 0) == (len(modified_segment_list) == 0):
            warnings.warn("There should be none images without any segment.")
        if len(modified_segment_list) == 0:
            segment_class_ids_and_masks = []

        segment_masks = []
        is_fgs = []
        modified_segments = []
        retrieval_segments_rgb = []
        retrieval_segment_masks = []
        points_before_moved = []
        points_after_moved = []
        warped = []

        if len(segment_class_ids_and_masks) == 0:  # Doesn't match any segment.
            segment_masks = [torch.zeros(1, int(self.opt.crop_size / self.opt.aspect_ratio), self.opt.crop_size)]
            is_fgs = [torch.Tensor([False]).bool()]
            modified_segments = [torch.zeros(3, int(self.opt.crop_size / self.opt.aspect_ratio), self.opt.crop_size) - 1]
            retrieval_segments_rgb = [torch.zeros(3, int(self.opt.crop_size / self.opt.aspect_ratio), self.opt.crop_size) - 1]
            retrieval_segment_masks = [torch.zeros(1, int(self.opt.crop_size / self.opt.aspect_ratio), self.opt.crop_size)]
            before_moved, after_moved, _ = warp_points(torch.zeros(1, int(self.opt.crop_size / self.opt.aspect_ratio), self.opt.crop_size))
            points_before_moved = [torch.from_numpy(before_moved)]
            points_after_moved = [torch.from_numpy(after_moved)]
            warped = [torch.Tensor([True]).bool()]  # set a dummy warp.

        for segment_mask, \
            is_fg, \
            modified_segment, \
            retrieval_segment_rgb, \
            retrieval_segment_mask in zip(
                                            segment_mask_list,
                                            is_fgs_list,
                                            modified_segment_list,
                                            retrieval_segment_rgb_list,
                                            retrieval_segment_mask_list
                                        ):

            tensor_mask = transform_label(to_pil(segment_mask)) * 255.

            segment_masks.append(tensor_mask)

            is_fgs.append(torch.Tensor([is_fg]).bool())

            modified_segments.append(transform_image(to_pil(modified_segment)))

            retrieval_segments_rgb.append(transform_image(to_pil(retrieval_segment_rgb)))

            retrieval_segment_masks.append(transform_label(to_pil(retrieval_segment_mask)) * 255.)

            before_moved, after_moved, is_warped = warp_points(tensor_mask)
            if is_warped:
                points_before_moved.append(torch.from_numpy(before_moved))
                points_after_moved.append(torch.from_numpy(after_moved))
            warped.append(torch.Tensor([is_warped]).bool())

        if len(points_before_moved) == 0:  # for images without segments to warp, we DO 1 time of dummy warping.
            before_moved, after_moved, _ = warp_points(torch.zeros(1, int(self.opt.crop_size / self.opt.aspect_ratio), self.opt.crop_size))
            points_before_moved = [torch.from_numpy(before_moved)]
            points_after_moved = [torch.from_numpy(after_moved)]
            warped[0] = torch.Tensor([True]).bool()  # warp the first segment.

        segment_masks = torch.stack(segment_masks, dim=0)
        is_fgs = torch.cat(is_fgs, dim=0)
        modified_segments = torch.stack(modified_segments, dim=0)
        retrieval_segments_rgb = torch.stack(retrieval_segments_rgb, dim=0)
        retrieval_segment_masks = torch.stack(retrieval_segment_masks, dim=0)
        points_before_moved = torch.cat(points_before_moved, dim=0)
        points_after_moved = torch.cat(points_after_moved, dim=0)
        warped = torch.cat(warped)
        data_length = segment_masks.shape[0]

        parameters = [
            segment_masks,
            is_fgs,
            modified_segments,
            retrieval_segments_rgb,
            retrieval_segment_masks,
            points_before_moved,
            points_after_moved,
            warped,
            torch.Tensor([data_length])
        ]
        return parameters


class ADE20kPreprocessor(BasePreprocessor):
    def __init__(self, opt, preserved_ratio=0.9):
        super().__init__(preserved_ratio, opt=opt)

        # Download from:
        # `https://raw.githubusercontent.com/CSAILVision/placeschallenge/master/instancesegmentation/categoryMapping.txt`
        category_mapping_path = os.path.join(opt.dataroot, "categoryMapping.txt")
        try:
            instance_id2class_id = np.loadtxt(category_mapping_path, dtype=str, skiprows=1, usecols=[0, 1])
        except IOError:
            print('Downloading `categoryMapping.txt`...')
            import wget
            url = 'https://raw.githubusercontent.com/CSAILVision/placeschallenge/master/instancesegmentation/categoryMapping.txt'
            wget.download(url, category_mapping_path)
            print('Done.')
            instance_id2class_id = np.loadtxt(category_mapping_path, dtype=str, skiprows=1, usecols=[0, 1])
        self.instance_id2class_id = {int(a[0]): int(a[1]) for a in instance_id2class_id}


    def parse_segments(self, ade20k_instance_label, ade20k_class_label, drop_out=0):
        """
        Parse segments in ADE20k dataset.
        :param ade20k_instance_label: Instance ids provided in ADE20k.
        :param ade20k_class_label: Class ids provided in ADE20k.
        :param drop_out: Prob. to drop a b.g. segmentation out.
        :return: Class id and its correspond mask.
        """

        fg_class_ids = self.instance_id2class_id.values()
        id_and_mask = []
        # for instance label.
        Om = ade20k_instance_label[:, :, 0]
        Oi = ade20k_instance_label[:, :, 1]
        for instance_index in np.unique(Oi):
            if instance_index == 0:
                continue
            mask = instance_index == Oi
            instance_id = Om[mask][0]
            class_id = self.instance_id2class_id[instance_id]
            # if random.uniform(0, 1) >= drop_out:
            id_and_mask.append((class_id, mask))

        # for class label.
        for class_id in np.unique(ade20k_class_label):
            if class_id == 0:
                continue
            if class_id not in fg_class_ids:
                all_mask = class_id == ade20k_class_label
                segments_masks = parse_connected_components_mask(all_mask)
                for mask in segments_masks:
                    if random.uniform(0, 1) >= drop_out:
                        id_and_mask.append((class_id, mask))

        id_and_mask = sort_segments_by_area(id_and_mask, preserved_ratio=self.preserved_ratio)
        return id_and_mask

    def preprocess_data(self, ade20k_instance_label, ade20k_class_label, image_pil, transform_image, transform_label,
                        drop_out=0):
        out = super()._process_data(
            ade20k_instance_label=ade20k_instance_label,
            ade20k_class_label=ade20k_class_label,
            image_pil=image_pil,
            fg_mask_id_list=self.instance_id2class_id.values(),
            transform_label=transform_label,
            transform_image=transform_image,
            drop_out=drop_out
        )
        return out

class ADE20kOutdoorPreprocessor(ADE20kPreprocessor):
    def __init__(self, opt):
        super().__init__(opt)
        category_mapping_path = os.path.join(opt.data_root, "categoryMapping.txt")
        instance_id2class_id = np.loadtxt(category_mapping_path, dtype=str, skiprows=1, usecols=[0, 1])
        self.instance_id2class_id = {int(a[0]): int(a[1]) for a in instance_id2class_id}



class CityscapesPreprocessor(BasePreprocessor):

    def __init__(self, opt, preserved_ratio=0.9):
        super().__init__(preserved_ratio, opt=opt)
        self.fg_mask_id_list = [i for i in range(24, 34)]

    def parse_segments(self, cityscapes_instance_label, drop_out=0):
        """
        Parse segments in Cityscapes dataset.
        :param drop_out:
        :param cityscapes_instance_label: Instance ids provided in Cityscapes.
        :return: Class id and its correspond mask.
        """
        id_and_mask = []
        for label in np.unique(cityscapes_instance_label):
            if label < 100:  # class label which less than class label numbers < 34 < 100 and instance id > 1000
                class_id = label
                mask = label == cityscapes_instance_label
                segment_masks = parse_connected_components_mask(mask)
                for segment_mask in segment_masks:
                    id_and_mask.append((class_id, segment_mask))

            else:
                segment_mask = label == cityscapes_instance_label
                class_id = label // 1000
                id_and_mask.append((class_id, segment_mask))

        return sort_segments_by_area(id_and_mask, preserved_ratio=self.preserved_ratio, drop_out=drop_out)

    def preprocess_data(self, cityscapes_instance_label, image_pil, transform_image, transform_label, drop_out=0):
        out = super()._process_data(
            cityscapes_instance_label=cityscapes_instance_label,
            image_pil=image_pil,
            fg_mask_id_list=self.fg_mask_id_list,
            transform_label=transform_label,
            transform_image=transform_image,
            drop_out=drop_out
        )
        return out


class COCOPreprocessor(BasePreprocessor):
    def __init__(self, opt, preserved_ratio=0.9):

        self.instance_class_id2class_name = {}
        # todo change these codes
        with open(settings.instance_train, 'r') as f:
            train2017_json = json.load(f)
        for item in train2017_json["categories"]:
            self.instance_class_id2class_name[item["id"]] = item["name"]

        super().__init__(preserved_ratio, opt=opt)

    def parse_segments(self, instance_label, class_label, drop_out=0):
        ids_and_masks = []
        # for f.g. instance
        fg_cls_lst = []
        for instance_index in np.unique(instance_label):
            if instance_index != 255:
                mask = instance_label == instance_index
                labels = class_label[mask]
                label_id = np.argmax(np.bincount(labels)) + 1  # label start from ZERO so add ONE
                fg_cls_lst.append(label_id)
                if label_id != 256:
                    ids_and_masks.append([label_id, mask])
        # for b.g. things
        for label_id in np.unique(class_label):
            if label_id != 255 and label_id + 1 not in fg_cls_lst:
                class_mask = class_label == label_id
                masks = parse_connected_components_mask(class_mask)
                for mask in masks:
                    ids_and_masks.append([label_id + 1, mask])  # label start from ZERO so add ONE

        return sort_segments_by_area(ids_and_masks, preserved_ratio=self.preserved_ratio, drop_out=drop_out)

    def preprocess_data(self, instance_label, class_label, image_pil, transform_image, transform_label, drop_out=0):
        out = super()._process_data(
            instance_label=instance_label,
            class_label=class_label,
            image_pil=image_pil,
            fg_mask_id_list=self.instance_class_id2class_name.keys(),
            transform_label=transform_label,
            transform_image=transform_image,
            drop_out=drop_out
        )
        return out


# functions.
def parse_connected_components_mask(binary_mask):
    all_mask255 = binary_mask.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        all_mask255,
        connectivity=8,
        ltype=None
    )
    masks = [labels == component_id for component_id in range(1, num_labels)]
    return masks


def sort_segments_by_area(id_and_mask, preserved_ratio, drop_out=0):
    each_mask_id_and_masks = {}
    for class_id, imask in id_and_mask:

        if class_id not in each_mask_id_and_masks.keys():
            each_mask_id_and_masks[class_id] = [imask]
        else:
            each_mask_id_and_masks[class_id].append(imask)
    results = []
    for k, v in each_mask_id_and_masks.items():
        each_mask_id_and_masks[k].sort(key=lambda mask: mask.sum())
        each_mask_id_and_masks[k].reverse()
        all_area = sum([m.sum() for m in each_mask_id_and_masks[k]])
        retain_area = all_area * preserved_ratio
        select_masks = []

        select_masks_area = 0
        for segment_mask in each_mask_id_and_masks[k]:
            if select_masks_area < retain_area:
                select_masks.append(segment_mask)
            select_masks_area = select_masks_area + segment_mask.sum()

        for mask in select_masks:
            if random.uniform(0, 1) >= drop_out:
                results.append((k, mask))

    return results


def warp_points(img_mask, num_samples=10, sample_shift_points_number=3, warp_ratio=0.22):
    """
    Calculate warping control points according to a binary mask.
    First, search the edge of mask. Second choose `num_samples` points from the edge at equal intervals.
    Third, choose `sample_shift_points_number` points randomly from the chosen points and move them a random length,
    which sample from a uniform distribution: `[- sqrt {mask_area} * warp_ratio, + sqrt{mask_area} * warp_ratio]`
    :param img_mask: binary mask.
    :param num_samples: number of points which chosen from edge.
    :param sample_shift_points_number: how many points sample from the chosen points.
    :param warp_ratio: control the degree of warping.
    :return: warp from (y, x) , warp to (y', x'), warp or not. Shapes of 1 x `num_samples` x 2
    """

    # define a dummy warp.
    def dummy_warp():
        ys = np.arange(num_samples - 1)
        ys = ys.reshape((1, -1, 1))
        xs = np.zeros(shape=(1, num_samples - 1, 1))
        coor = np.concatenate([xs, ys], axis=-1)
        coor = np.concatenate([coor, np.array([[[1, 1]]])], axis=1)
        return coor, coor, False

    def swap_last_dim(swap_points):
        points_ = np.zeros_like(swap_points)
        points_[..., 0] = swap_points[..., 1]
        points_[..., 1] = swap_points[..., 0]
        return points_

    def checkpoints(points_to_check):
        points_to_check = points_to_check[0]
        num = points_to_check.shape[0]
        ones = np.ones(shape=(num, 1))
        points_cat_ones = np.concatenate([points_to_check, ones], axis=-1)
        rank = np.linalg.matrix_rank(points_cat_ones)
        return rank == 3

    def check_repeat_points(points):
        points = points[0]
        num = points.shape[0]
        for i in range(num - 1):
            for j in range(i + 1, num):
                if np.abs(np.abs(points[i]) - np.abs(points[j])).sum() == 0:  # repeat point
                    return False
        return True

    img_mask = np.array(img_mask.squeeze())
    mask_area = math.sqrt(np.sum(img_mask))
    warp_size = mask_area * warp_ratio

    contours, hierarchy = cv2.findContours(img_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return dummy_warp()

    # find the contour with most points.
    contour_with_max_points_number_index = 0
    points_number = len(contours[contour_with_max_points_number_index])
    if len(contours) > 1:
        for contour_index, contour in enumerate(contours[1:]):

            if len(contour) > points_number:
                contour_with_max_points_number_index = contour_index + 1
                points_number = len(contour)

    contours_points_number = len(contours[contour_with_max_points_number_index])

    if contours_points_number // (num_samples + 1) == 0:
        return dummy_warp()

    selected_points = contours[
        contour_with_max_points_number_index
    ][
        np.linspace(0, contours_points_number - 1, num_samples, dtype=np.int)
    ]

    moving_point_index = np.random.choice(np.arange(0, num_samples), sample_shift_points_number)
    point_before_moved = np.reshape(selected_points.copy(), (1, -1, 2))

    selected_points[moving_point_index] = selected_points[moving_point_index].astype(np.float) + \
                                          np.clip(
                                              np.random.uniform(-0.4, 0.4, size=(sample_shift_points_number, 1, 2)),
                                              -0.4,
                                              0.4) * warp_size

    point_after_moved = np.reshape(selected_points.copy(), (1, -1, 2)).astype(np.int)
    if not checkpoints(point_after_moved):
        return dummy_warp()

    if not check_repeat_points(point_after_moved):
        return dummy_warp()

    return swap_last_dim(point_before_moved), swap_last_dim(point_after_moved), True


def preprocess_batch(batch_data, save_flag, src_dir, dest_dir):
    if "parameters" not in batch_data.keys():
        return batch_data
    else:
        if save_flag:
            try:
                for p in batch_data['image_path']:
                    src_img_p = p
                    save_p_modified = replace_path(src_img_p.replace('jpg', 'png'), src_dir, dest_dir, "modified")
                    save_p_retrieval = replace_path(src_img_p.replace('jpg', 'png'), src_dir, dest_dir, 'retrieval')
                    Image.open(save_p_modified)
                    Image.open(save_p_retrieval)
                return None
            except :
                pass

        batch_data = copy.deepcopy(batch_data)
        parameter = batch_data['parameters']
        composed = compose_retrieval_and_modified_images(parameter)
        # cache files
        if save_flag:
            im_paths = batch_data['image_path']
            modified = composed["modified_image"]
            retrieval = composed['retrieval_image']
            for i in range(len(im_paths)):
                src_img_p = im_paths[i]

                # for modified
                save_p_modified = replace_path(src_img_p.replace('jpg', 'png'), src_dir, dest_dir, "modified")
                modified_image = modified[i].cpu()
                cache_image(modified_image, save_p_modified)

                # for retrieval
                save_p_retrieval = replace_path(src_img_p.replace('jpg', 'png'), src_dir, dest_dir, 'retrieval')
                retrieval_image = retrieval[i].cpu()
                cache_image(retrieval_image, save_p_retrieval)

        del batch_data['parameters']
        batch_data.update(composed)
        return batch_data


def cache_image(tensor, save_p):
    numpy_image = tensor.numpy()
    numpy_image = (np.transpose(numpy_image, (1, 2, 0)) + 1) / 2.0 * 255.0
    numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)
    Image.fromarray(numpy_image).save(save_p)


def replace_path(path, src_path, *dest_path):
    dest_paths = os.path.join(*dest_path)
    p = path.replace(src_path, dest_paths)
    if not os.path.isdir(os.path.dirname(p)):
        os.makedirs(os.path.dirname(p))
    return p


