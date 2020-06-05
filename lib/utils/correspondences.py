import numpy as np
import torch
import sys
from fcn.config import cfg
import torch.nn.functional as F

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor


def flattened_pixel_locations_to_u_v(flat_pixel_locations, image_width):
    """
    :param flat_pixel_locations: A torch.LongTensor of shape torch.Shape([n,1]) where each element
     is a flattened pixel index, i.e. some integer between 0 and 307,200 for a 640x480 image
    :type flat_pixel_locations: torch.LongTensor
    :return A tuple torch.LongTensor in (u,v) format
    the pixel and the second column is the v coordinate
    """
    return (flat_pixel_locations%image_width, flat_pixel_locations/image_width)


def flatten_uv_tensor(uv_tensor, image_width):
    return uv_tensor[1].long() * image_width + uv_tensor[0].long()


def where(cond, x_1, x_2):
    cond = cond.type(dtype_float)
    return (cond * x_1) + ((1 - cond) * x_2)


def rand_select_pixel(width, height, num_samples=1):
    two_rand_numbers = torch.rand(2, num_samples)
    two_rand_numbers[0, :] = two_rand_numbers[0, :] * width
    two_rand_numbers[1, :] = two_rand_numbers[1, :] * height
    two_rand_ints = torch.floor(two_rand_numbers).type(dtype_long)
    return two_rand_ints


def random_sample_from_masked_image_torch(img_mask, num_samples):
    image_height, image_width = img_mask.shape

    if isinstance(img_mask, np.ndarray):
        img_mask_torch = torch.from_numpy(img_mask).float()
    else:
        img_mask_torch = img_mask

    # This code would randomly subsample from the mask
    mask = img_mask_torch.view(image_width * image_height, 1).squeeze(1)
    mask_indices_flat = torch.nonzero(mask)
    if len(mask_indices_flat) == 0:
        return (None, None)

    rand_numbers = torch.rand(num_samples) * len(mask_indices_flat)
    rand_indices = torch.floor(rand_numbers).long().type(dtype_long)
    uv_vec_flattened = torch.index_select(mask_indices_flat, 0, rand_indices).squeeze(1)
    uv_vec = flattened_pixel_locations_to_u_v(uv_vec_flattened, image_width)
    return uv_vec


def create_non_matches(uv_a, uv_b_non_matches, multiplier):
    uv_a_long = (torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
                 torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1))

    uv_b_non_matches_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))

    return uv_a_long, uv_b_non_matches_long


def find_correspondences(features_a, features_b, label_a, label_b, num_samples=1):
    metric = cfg.TRAIN.EMBEDDING_METRIC
    if len(label_a.shape) == 4:
        mask_a = label_a[0, 1, : , :].cpu()
    elif len(label_a.shape) == 3:
        mask_a = label_a[0, : , :].cpu()
    else:
        mask_a = label_a.cpu()
    width_b = features_b.shape[3]

    # sample pixels
    uv_a = random_sample_from_masked_image_torch(mask_a, num_samples=num_samples)
    descriptor = features_a[:, :, uv_a[1], uv_a[0]].unsqueeze(2)
    features_b = features_b.view(features_b.shape[0], features_b.shape[1], -1).unsqueeze(3)
    if metric == 'euclidean':
        norm_degree = 2
        distances = (features_b - descriptor).norm(norm_degree, 1).squeeze(0)
    elif metric == 'cosine':
        distances = 0.5 * (1 - (features_b * descriptor).sum(dim=1)).squeeze(0)
    index = torch.argmin(distances, dim=0).cpu()
    uv_b = flattened_pixel_locations_to_u_v(index, width_b)

    # top_values, top_indexes = torch.topk(distances, k=10, dim=0, largest=False)
    return uv_a, uv_b


def compute_prototype_distances(features_a, features_b, label_a):
    metric = cfg.TRAIN.EMBEDDING_METRIC
    if len(label_a.shape) == 4:
        mask_a = label_a[:, 1, : , :]
        mask_a = mask_a.unsqueeze(1)
    elif len(label_a.shape) == 3:
        mask_a = label_a
        mask_a = mask_a.unsqueeze(0)
    else:
        mask_a = label_a
        mask_a = mask_a.unsqueeze(0).unsqueeze(0)

    prototypes = torch.sum(features_a * mask_a, dim=[2, 3]) / (torch.sum(mask_a, dim=[2, 3]) + 1e-10)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    # compute distances
    tiled_prototypes = prototypes.unsqueeze(2).unsqueeze(3)
    if metric == 'euclidean':
        norm_degree = 2
        distances = (features_b - tiled_prototypes).norm(norm_degree, 1)
    elif metric == 'cosine':
        distances = 0.5 * (1 - (features_b * tiled_prototypes).sum(dim=1))
    return distances


def create_non_correspondences(uv_b_matches, img_b_shape, num_non_matches_per_match=5, img_b_mask=None):
    image_width = img_b_shape[1]
    image_height = img_b_shape[0]
    num_matches = len(uv_b_matches[0])

    def get_random_uv_b_non_matches():
        return rand_select_pixel(width=image_width, height=image_height,
                                 num_samples=num_matches*num_non_matches_per_match)

    if img_b_mask is not None:
        img_b_mask_flat = img_b_mask.view(-1, 1).squeeze(1)
        mask_b_indicies_flat = torch.nonzero(img_b_mask_flat)
        if len(mask_b_indicies_flat) == 0:
            print("Warning, empty mask b")
            uv_b_non_matches = get_random_uv_b_non_matches()
        else:
            num_samples = num_matches * num_non_matches_per_match
            rand_numbers_b = torch.rand(num_samples) * len(mask_b_indicies_flat)
            rand_indicies_b = torch.floor(rand_numbers_b).long()
            randomized_mask_b_indicies_flat = torch.index_select(mask_b_indicies_flat, 0, rand_indicies_b).squeeze(1)
            uv_b_non_matches = (randomized_mask_b_indicies_flat % image_width, randomized_mask_b_indicies_flat / image_width)
    else:
        uv_b_non_matches = get_random_uv_b_non_matches()

    uv_b_non_matches = (uv_b_non_matches[0].view(num_matches, num_non_matches_per_match),
                        uv_b_non_matches[1].view(num_matches, num_non_matches_per_match))

    copied_uv_b_matches_0 = torch.t(uv_b_matches[0].repeat(num_non_matches_per_match, 1))
    copied_uv_b_matches_1 = torch.t(uv_b_matches[1].repeat(num_non_matches_per_match, 1))

    diffs_0 = copied_uv_b_matches_0 - uv_b_non_matches[0].type(dtype_float)
    diffs_1 = copied_uv_b_matches_1 - uv_b_non_matches[1].type(dtype_float)

    diffs_0_flattened = diffs_0.reshape(-1, 1)
    diffs_1_flattened = diffs_1.reshape(-1, 1)

    diffs_0_flattened = torch.abs(diffs_0_flattened).squeeze(1)
    diffs_1_flattened = torch.abs(diffs_1_flattened).squeeze(1)

    need_to_be_perturbed = torch.zeros_like(diffs_0_flattened)
    ones = torch.ones_like(diffs_0_flattened)
    num_pixels_too_close = 10.0
    threshold = torch.ones_like(diffs_0_flattened) * num_pixels_too_close

    need_to_be_perturbed = where(diffs_0_flattened < threshold, ones, need_to_be_perturbed)
    need_to_be_perturbed = where(diffs_1_flattened < threshold, ones, need_to_be_perturbed)

    minimal_perturb = num_pixels_too_close / 2
    minimal_perturb_vector = (torch.rand(len(need_to_be_perturbed)) * 2).floor() * (minimal_perturb * 2) - minimal_perturb
    std_dev = 10
    random_vector = torch.randn(len(need_to_be_perturbed)) * std_dev + minimal_perturb_vector
    perturb_vector = need_to_be_perturbed * random_vector

    uv_b_non_matches_0_flat = uv_b_non_matches[0].view(-1, 1).type(dtype_float).squeeze(1)
    uv_b_non_matches_1_flat = uv_b_non_matches[1].view(-1, 1).type(dtype_float).squeeze(1)

    uv_b_non_matches_0_flat = uv_b_non_matches_0_flat + perturb_vector
    uv_b_non_matches_1_flat = uv_b_non_matches_1_flat + perturb_vector

    lower_bound = 0.0
    upper_bound = image_width * 1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * upper_bound

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat > upper_bound_vec,
                                    uv_b_non_matches_0_flat - upper_bound_vec,
                                    uv_b_non_matches_0_flat)

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat < lower_bound_vec,
                                    uv_b_non_matches_0_flat + upper_bound_vec,
                                    uv_b_non_matches_0_flat)

    lower_bound = 0.0
    upper_bound = image_height * 1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * upper_bound

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat > upper_bound_vec,
                                    uv_b_non_matches_1_flat - upper_bound_vec,
                                    uv_b_non_matches_1_flat)

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat < lower_bound_vec,
                                    uv_b_non_matches_1_flat + upper_bound_vec,
                                    uv_b_non_matches_1_flat)

    return (uv_b_non_matches_0_flat.view(num_matches, num_non_matches_per_match),
            uv_b_non_matches_1_flat.view(num_matches, num_non_matches_per_match))
