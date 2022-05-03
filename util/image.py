import collections
import math

from imageio import imread
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

import settings
from models.networks import tps


def to_box(mask, bounding_box=True):
    """
    Return a bbox from binary mask.
    :param bounding_box: whether return bounding box.
    :param mask: binary mask.
    :return: [x y w h].
    """
    as_row = np.sum(mask, axis=0)
    mask_row = np.nonzero(as_row)
    as_col = np.sum(mask, axis=1)
    mask_col = np.nonzero(as_col)
    h1 = np.min(mask_col)
    h2 = np.max(mask_col) + 1
    w1 = np.min(mask_row)
    w2 = np.max(mask_row) + 1
    if not bounding_box:
        return h1, h2, w1, w2
    else:
        x = w1
        y = h1
        w = w2 - w1
        h = h2 - h1
        return x, y, w, h


def saveim(img, save_pth, new_size=None, method=Image.BICUBIC):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1) / 2. * 255.
        img = img.astype(np.uint8)
        img = img.squeeze()

    PILimg = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    if PILimg.mode == 'F':
        PILimg = PILimg.convert('RGB')
    if new_size != None:
        PILimg = PILimg.resize(new_size, method)
    PILimg.save(save_pth)


def grid_img_result(imgs, cols=4, rows=None, line=1, line_color=(255, 255, 255), img_shape=None):
    """
    Show a series of images in grid.
    :param imgs: can be a `tensor`, `ndarray`, and list of above.
    :param cols: How many column.
    :param rows: How many rows.
    :param line: Inter line size.
    :param line_color: Inter line color.
    :param img_shape: Shape of the output image.
    :return: Grid image.
    """
    if isinstance(imgs, list):
        img_list = []
        pre_size = None
        for img in imgs:
            if isinstance(img, (torch.Tensor, np.ndarray)):
                size = img.shape[-3:]
                if pre_size is not None:
                    assert pre_size == size, f"Input img represent should be same shape, bug get " \
                                             f"{pre_size} and {size}."
                pre_size = size
        for img in imgs:
            if isinstance(img, str):
                img = openim(img, img_shape)
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img_list.append(img.cpu())
        imgs = torch.cat(img_list, dim=0)
    n, c, h, w = imgs.shape
    rows = rows if rows else math.ceil(n / cols)
    img_row_lst = []
    if c == 3:
        color_factor = np.reshape(np.array(line_color), newshape=(1, 1, 3)).astype(np.uint8)
    else:
        color_factor = 1
    for i in range(rows):
        img_col_lst = []
        for j in range(cols):
            if i * cols + j < n:
                img_col_lst.append(tensor2im(imgs[i * cols + j]))
            else:
                img_col_lst.append(np.zeros(shape=(h, w, c), dtype=np.uint8))
            if line > 0 and j < cols - 1:
                img_col_lst.append(np.ones(shape=(h, line, c), dtype=np.uint8) * color_factor)
        img_row = np.concatenate(img_col_lst, axis=1)
        img_row_lst.append(img_row)
        if line > 0 and i < rows - 1:
            img_row_lst.append(
                np.ones(shape=(line, w * cols + line * (cols - 1), c), dtype=np.uint8) * color_factor)
    img = np.concatenate(img_row_lst, axis=0)
    img = img.squeeze()
    pil_img = Image.fromarray(img)

    return pil_img


def openim(img_path, new_size=None) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    if new_size is None:
        new_size = img.size
    transform_list_img = [transforms.Resize(new_size, transforms.InterpolationMode.BICUBIC), transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list_img)(img)


def tensor2im(tensor):
    if tensor.ndim == 4:
        assert tensor.shape[0] == 1, "Image tensor should have the shape of (1, c, w, h) or (c, w, h)"
        tensor = tensor[0]
    if isinstance(tensor, torch.Tensor):
        img = tensor.cpu().numpy()
    else:
        img = tensor
    img = np.transpose(img, (1, 2, 0))
    img = (img + 1) / 2. * 255.
    img = img.astype(np.uint8)
    return img


def _RBG2LAB(image_in_rgb_space):
    EPSILON = 1e-10

    RGB2LMS_MAT = np.array([[0.3811, 0.5783, 0.0402],
                            [0.1967, 0.7244, 0.0782],
                            [0.0241, 0.1288, 0.8444]])

    LMS2LAB_MAT = np.array([[1 / np.sqrt(3), 0, 0],
                            [0, 1 / np.sqrt(6), 0],
                            [0, 0, 1 / np.sqrt(2)]]) @ \
                  np.array([[1, 1, 1],
                            [1, 1, -2],
                            [1, -1, 0]])

    image_in_rgb_space = np.expand_dims(image_in_rgb_space.astype(np.float), axis=-1)
    image_in_lms_space = np.log(np.maximum(RGB2LMS_MAT @ image_in_rgb_space, EPSILON))
    image_in_lab_space = LMS2LAB_MAT @ image_in_lms_space
    return image_in_lab_space.squeeze(-1)


def _LAB2RGB(image_in_lab_space):
    image_in_lab_space = np.expand_dims(image_in_lab_space, axis=-1)
    LAB2LMS_MAT = np.array([[1, 1, 1],
                            [1, 1, -1],
                            [1, -2, 0]]) @ \
                  np.array([[1 / np.sqrt(3), 0, 0],
                            [0, 1 / np.sqrt(6), 0],
                            [0, 0, 1 / np.sqrt(2)]])
    image_in_lms_space = LAB2LMS_MAT @ image_in_lab_space
    image_in_lms_space = np.exp(image_in_lms_space)
    LMS2RGB_MAT = np.array([[4.4679, -3.5873, 0.1193],
                            [-1.2186, 2.3809, -0.1624],
                            [0.0497, -0.2439, 1.2045]])
    image_in_rgb_space = LMS2RGB_MAT @ image_in_lms_space
    return np.clip(image_in_rgb_space.squeeze(-1), 0, 255).astype(np.uint8)


def region_color_transfer(src_image, src_region_mask, dest_image, dest_region_mask):
    EPSILON = 1e-10

    src_region_mask = src_region_mask.astype(np.bool)
    dest_region_mask = dest_region_mask.astype(np.bool)
    result = src_image.copy()

    src_image = src_image[src_region_mask]
    dest_image = dest_image[dest_region_mask]

    src_image_in_lab_space = _RBG2LAB(src_image)
    dest_image_in_lab_space = _RBG2LAB(dest_image)
    src_image_in_lab_mean_per_channel = np.mean(np.reshape(src_image_in_lab_space, (-1, 3)), axis=0, keepdims=True)
    src_image_in_lab_std_per_channel = np.std(np.reshape(src_image_in_lab_space, (-1, 3)), axis=0, keepdims=True)
    dest_image_in_lab_mean_per_channel = np.mean(np.reshape(dest_image_in_lab_space, (-1, 3)), axis=0, keepdims=True)
    dest_image_in_lab_std_per_channel = np.std(np.reshape(dest_image_in_lab_space, (-1, 3)), axis=0, keepdims=True)

    src_image_in_lab_space = src_image_in_lab_space - src_image_in_lab_mean_per_channel
    src_image_in_lab_space = src_image_in_lab_space * np.maximum(dest_image_in_lab_std_per_channel,
                                                                 EPSILON) / np.maximum(src_image_in_lab_std_per_channel,
                                                                                       EPSILON)
    src_image_in_lab_space = src_image_in_lab_space + dest_image_in_lab_mean_per_channel
    src_image_with_dest_image_color_feature = _LAB2RGB(src_image_in_lab_space)
    result[src_region_mask] = src_image_with_dest_image_color_feature

    return result


def _RGB2HLS(image_in_rgb_space):
    image_in_rgb_space = image_in_rgb_space.astype(np.float) / 255.
    max_ = np.max(image_in_rgb_space, axis=1)
    min_ = np.min(image_in_rgb_space, axis=1)
    r = image_in_rgb_space[..., 0]
    g = image_in_rgb_space[..., 1]
    b = image_in_rgb_space[..., 2]
    l = (max_ + min_) / 2.
    h = np.zeros_like(l)
    s = np.zeros_like(l)

    condition = max_ == min_
    h[condition] = 0

    condition = (max_ == r) & (g > b) & (max_ != min_)
    h[condition] = 60 * (g[condition] - b[condition]) / (max_[condition] - min_[condition])

    condition = (max_ == r) & (g < b) & (max_ != min_)
    h[condition] = 60 * (g[condition] - b[condition]) / (max_[condition] - min_[condition]) + 360

    condition = (max_ == g) & (max_ != min_)
    h[condition] = 60 * (b[condition] - r[condition]) / (max_[condition] - min_[condition]) + 120

    condition = (max_ == b) & (max_ != min_)
    h[condition] = 60 * (r[condition] - g[condition]) / (max_[condition] - min_[condition]) + 240

    condition = (l == 0) | (max_ == min_)
    s[condition] = 0

    condition = (0 < l) & (l <= 0.5)
    s[condition] = (max_[condition] - min_[condition]) / (max_[condition] + min_[condition])

    condition = (l > 0.5) & (l < 1)
    s[condition] = (max_[condition] - min_[condition]) / (2 - (max_[condition] + min_[condition]))

    return np.stack([h, l, s], axis=-1)


def region_light_transfer(src_image, src_region_mask, dest_image, dest_region_mask):
    src_image = src_image.astype(np.float)
    dest_image = dest_image.astype(np.float)

    src_region_mask = src_region_mask.astype(np.bool)
    dest_region_mask = dest_region_mask.astype(np.bool)
    result = src_image.copy()

    src_image = src_image[src_region_mask]
    dest_image = dest_image[dest_region_mask]

    src_image_in_hls_space = _RGB2HLS(src_image)
    dest_image_in_hls_space = _RGB2HLS(dest_image)

    src_light = np.mean(src_image_in_hls_space[..., 1])
    dest_light = np.mean(dest_image_in_hls_space[..., 1])

    src_image = src_image * dest_light / np.maximum(src_light, 1e-10)
    src_image = np.clip(src_image, 0, 255)

    result[src_region_mask] = src_image
    return result.astype(np.uint8)


def compose_retrieval_and_modified_images(parameter, batch_size=32):
    """
    return:
     A batch of images from  warped instances and retrieval instances.
    """
    parameter_ = [p.cuda(settings.calculating_gpu, non_blocking=True) for p in parameter]

    instance_masks, \
    is_fgs, \
    modified_instances, \
    retrieval_instance_rgbs, \
    retrieval_instance_masks, \
    points_before_shift, \
    points_after_shift, \
    warped, \
    data_length = parameter_

    instance_to_warp = []
    mask_to_warp = []
    for i, is_warped in enumerate(warped):
        if is_warped:
            instance_to_warp.append(modified_instances[i])
            mask_to_warp.append(instance_masks[i])

    # warp

    def mini_batch(*args):
        returns = []
        for arg in args:
            returns.append(torch.split(arg, batch_size))
        return returns

    batch_instance_to_warp, batch_points_before_shift, batch_points_after_shift, batch_mask_to_warp = mini_batch(
        torch.stack(instance_to_warp).cuda(settings.calculating_gpu, non_blocking=True) + 1,
        points_before_shift,
        points_after_shift,
        torch.stack(mask_to_warp, dim=0).cuda(settings.calculating_gpu, non_blocking=True).double()
    )

    warped_instances = []
    dense_flows = []
    warped_instance_masks = []
    for arg1, arg2, arg3, arg4 in zip(batch_instance_to_warp, batch_points_before_shift, batch_points_after_shift,
                                      batch_mask_to_warp):
        warped_ins, dense_flo = tps.spares_image_warp(
            arg1,
            arg2,
            arg3
        )
        warped_instance_m = F.grid_sample(arg4, dense_flo, mode='nearest')
        warped_instances.append(warped_ins)
        warped_instance_masks.append(warped_instance_m)
    warped_instances = torch.cat(warped_instances, dim=0)
    warped_instance_masks = torch.cat(warped_instance_masks, dim=0)

    warped_instances = warped_instances - 1
    warped_instances = iter(warped_instances)
    warped_instance_masks = iter(warped_instance_masks)

    warped_instances_ = []
    warped_instance_masks_ = []
    for i, is_warped in enumerate(warped):
        if is_warped:
            warped_instances_.append(next(warped_instances))
            warped_instance_masks_.append(next(warped_instance_masks))
        else:
            warped_instances_.append(modified_instances[i])
            warped_instance_masks_.append(instance_masks[i])

    split_list = [
        instance_masks,
        is_fgs,
        retrieval_instance_rgbs,
        retrieval_instance_masks,
        torch.stack(warped_instances_, dim=0).cuda(settings.calculating_gpu, non_blocking=True),
        torch.stack(warped_instance_masks_, dim=0).cuda(settings.calculating_gpu, non_blocking=True),
    ]

    retrieval_image = []
    modified_image = []
    for instance_masks_in_a_batch, \
        is_fgs_in_a_batch, \
        retrieval_instance_rgbs_in_a_batch, \
        retrieval_instance_masks_in_a_batch, \
        warped_instances_in_a_batch, \
        warped_instance_masks_in_a_batch \
            in zip(*(torch.split(p, data_length.cpu().int().numpy().tolist()) for p in split_list)):
        composed_retrieval_image = compose(
            instance_masks_in_a_batch,
            is_fgs_in_a_batch,
            retrieval_instance_rgbs_in_a_batch,
            retrieval_instance_masks_in_a_batch
        )

        retrieval_image.append(composed_retrieval_image)

        composed_modified_image = compose(
            instance_masks_in_a_batch,
            is_fgs_in_a_batch,
            warped_instances_in_a_batch,
            warped_instance_masks_in_a_batch
        )

        modified_image.append(composed_modified_image)

    return {
        "retrieval_image": torch.stack(retrieval_image, dim=0),
        "modified_image": torch.stack(modified_image, dim=0)
    }



def compose(
        instance_masks_in_a_batch,
        is_fgs_in_a_batch,
        retrieval_instance_rgbs_in_a_batch,
        retrieval_instance_masks_in_a_batch
):
    results = []

    assert len(instance_masks_in_a_batch) == len(is_fgs_in_a_batch)
    assert len(retrieval_instance_rgbs_in_a_batch) == len(retrieval_instance_masks_in_a_batch)
    assert len(is_fgs_in_a_batch) == len(retrieval_instance_rgbs_in_a_batch)
    for i in range(instance_masks_in_a_batch.shape[0]):

        instance_mask, is_fg, retrieval_instance, retrieval_mask = instance_masks_in_a_batch[i], \
                                                                   is_fgs_in_a_batch[i], \
                                                                   retrieval_instance_rgbs_in_a_batch[i], \
                                                                   retrieval_instance_masks_in_a_batch[i]
        if is_fg:

            all_fg_mask_except_this = []
            for j in range(instance_masks_in_a_batch.shape[0]):
                instance_mask_j, is_fg_j, retrieval_instance_j, retrieval_mask_j = instance_masks_in_a_batch[j], \
                                                                                   is_fgs_in_a_batch[j], \
                                                                                   retrieval_instance_rgbs_in_a_batch[
                                                                                       j], \
                                                                                   retrieval_instance_masks_in_a_batch[
                                                                                       j]
                if i != j:
                    if is_fg_j:
                        all_fg_mask_except_this.append(instance_mask_j)
            if len(all_fg_mask_except_this) == 0:
                illegal_mask = torch.zeros_like(instance_mask).cuda(1)
            else:
                illegal_mask = torch.sum(torch.stack(all_fg_mask_except_this, dim=0), dim=0)
            # illegal_mask = torch.unsqueeze(illegal_mask, dim=0)
            legal_mask = 1 - illegal_mask
            rgb = retrieval_instance + 1
            rgb = rgb * legal_mask
            results.append(rgb)

        else:

            all_retrieval_fg_mask = []
            for j in range(instance_masks_in_a_batch.shape[0]):
                instance_mask_j, is_fg_j, retrieval_instance_j, retrieval_mask_j = instance_masks_in_a_batch[j], \
                                                                                   is_fgs_in_a_batch[j], \
                                                                                   retrieval_instance_rgbs_in_a_batch[
                                                                                       j], \
                                                                                   retrieval_instance_masks_in_a_batch[
                                                                                       j]
                if is_fg_j:
                    all_retrieval_fg_mask.append(retrieval_mask_j.int())
            illegal_mask = 1 - instance_mask

            illegal_mask_list = [illegal_mask] + all_retrieval_fg_mask

            illegal_mask = torch.sum(torch.stack(illegal_mask_list, dim=0), dim=0)
            illegal_mask = illegal_mask > 0
            legal_mask = 1 - illegal_mask.int()

            # legal_mask = torch.unsqueeze(legal_mask, dim=0)

            rgb = retrieval_instance + 1

            rgb = rgb * legal_mask

            results.append(rgb)

        results.append((retrieval_instance + 1) * instance_mask)
    # print(len(results))

    added = torch.sum(torch.stack(results, dim=0), dim=0) - 1
    return added


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.image_pool = collections.OrderedDict()

    def query(self, image_p):
        if image_p in self.image_pool.keys():
            return self.image_pool[image_p]
        else:
            rgb = imread(image_p)
            self.image_pool[image_p] = rgb
            self.check_pool()
            return rgb

    def check_pool(self):
        if len(self.image_pool) > self.pool_size:
            first_key = list(self.image_pool.keys())[0]
            self.image_pool.pop(first_key)
            self.check_pool()


# functions.
def to_pil(image):
    if isinstance(image, Image.Image):
        return image
    else:
        return Image.fromarray(image.astype(np.uint8))
