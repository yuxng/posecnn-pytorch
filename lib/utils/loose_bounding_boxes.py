import torch
import numpy as np
from fcn.config import cfg
from utils.mean_shift import mean_shift_smart_init

def compute_centroids_and_loose_bounding_boxes(out_vertex, out_label, extents, xyz_images, depth_mask, intrinsic_matrix):
    batch_size = out_vertex.shape[0]
    stacked_size = out_vertex.shape[1]
    height = out_vertex.shape[2]
    width = out_vertex.shape[3]
    nclasses = stacked_size / 3
    out_label_indices = out_label.unsqueeze(1).long()
    label_onehot = torch.zeros(out_vertex.shape[0], nclasses, out_vertex.shape[2], out_vertex.shape[3]).cuda()
    label_onehot.scatter_(1, out_label_indices, 1)

    extents_batch = extents.repeat(batch_size, 1, 1)
    extents_largest_dim = torch.sqrt((extents_batch * extents_batch).sum(dim=2))
    extents_largest_dim = extents_largest_dim.repeat(1, 3)
    extents_largest_dim = extents_largest_dim.reshape(batch_size, 3, nclasses)
    extents_largest_dim = extents_largest_dim.transpose(1, 2).reshape(batch_size, 3 * nclasses)

    label_onehot_tiled = label_onehot.repeat(1, 1, 3, 1).view(batch_size, -1, height, width)
    xyz_images = xyz_images.repeat(1, stacked_size / 3, 1, 1)
    mask_repeat = depth_mask.repeat(1, stacked_size, 1, 1)
    masked_labels_tiled = label_onehot_tiled * mask_repeat

    delta_centers = out_vertex * masked_labels_tiled * extents_largest_dim.unsqueeze(2).unsqueeze(2) * 0.5
    xyz_centers = xyz_images * masked_labels_tiled

    center_predictions = xyz_centers - delta_centers

    object_centers = torch.zeros(batch_size, nclasses * 3).cuda().float()
    for b in range(batch_size):
        for k in range(object_centers.shape[1]):
            valid_points = torch.masked_select(center_predictions[b, k, :, :], masked_labels_tiled[b, k, :, :].byte())
            if valid_points.shape[0]:
                med_value = torch.median(valid_points)
                object_centers[b, k] = med_value
            else:
                object_centers[b, k] = 0.0

    min_coords = object_centers - extents_largest_dim / 2.0
    max_coords = object_centers + extents_largest_dim / 2.0

    min_coords = min_coords.reshape(batch_size, nclasses, 3)
    max_coords = max_coords.reshape(batch_size, nclasses, 3)

    object_centers_reshape = object_centers.reshape(batch_size, nclasses, 3)

    zs = torch.clamp(object_centers_reshape[:, :, 2], min=0.001)

    x_mins = min_coords[:, :, 0]
    x_maxs = max_coords[:, :, 0]

    y_mins = min_coords[:, :, 1]
    y_maxs = max_coords[:, :, 1]

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]

    col_mins = x_mins / zs * fx + px
    col_maxs = x_maxs / zs * fx + px

    row_mins = y_mins / zs * fy + py
    row_maxs = y_maxs / zs * fy + py

    col_mins = torch.clamp(col_mins, min=0.0, max=width)
    row_mins = torch.clamp(row_mins, min=0.0, max=height)

    col_maxs = torch.clamp(col_maxs, min=0.0, max=width)
    row_maxs = torch.clamp(row_maxs, min=0.0, max=height)

    col_mins = col_mins.reshape(nclasses * batch_size, 1)
    row_mins = row_mins.reshape(nclasses * batch_size, 1)
    col_maxs = col_maxs.reshape(nclasses * batch_size, 1)
    row_maxs = row_maxs.reshape(nclasses * batch_size, 1)

    class_range = torch.arange(nclasses).float()
    class_range = class_range.repeat(batch_size)
    class_range = class_range.unsqueeze_(1).reshape(nclasses * batch_size, 1)

    batch_ids = torch.arange(batch_size)
    batch_ids = batch_ids.repeat(nclasses).unsqueeze(1)
    batch_ids = batch_ids.reshape(nclasses, batch_size)
    batch_ids = batch_ids.transpose(0, 1).reshape(nclasses * batch_size, 1).float()

    bounding_boxes = torch.cat((batch_ids.cuda(), class_range.cuda(), col_mins, row_mins, col_maxs, row_maxs), dim=1)
    object_centers = object_centers_reshape.squeeze(0)

    final_centers = torch.zeros(1, object_centers.shape[1]).cuda().float()
    final_boxes = torch.zeros(1, bounding_boxes.shape[1]).cuda().float()

    for c in range(bounding_boxes.shape[0]):
        batch_id = bounding_boxes[c, 0].long()
        class_id = bounding_boxes[c, 1].long()

        if class_id == 0:
            continue

        if not torch.nonzero(object_centers[c, :]).shape[0]:
            continue

        col_min = torch.clamp(bounding_boxes[c, 2], min=0.0, max=width).int()
        row_min = torch.clamp(bounding_boxes[c, 3], min=0.0, max=height).int()
        col_max = torch.clamp(bounding_boxes[c, 4], min=0.0, max=width).int()
        row_max = torch.clamp(bounding_boxes[c, 5], min=0.0, max=height).int()

        object_segmentation_image = label_onehot[batch_id, class_id, :, :]
        box_area = object_segmentation_image[row_min:row_max, col_min:col_max]

        total_pixels = torch.nonzero(box_area).shape[0]

        coverage = total_pixels * 100.0 / torch.clamp(((row_max - row_min) * (col_max - col_min)), min=1.0)

        if coverage <= 2.0:
            continue

        if col_max - col_min <= 2.0 / 100.0 * width or row_max - row_min <= 2.0 / 100.0 * height:
            continue

        local_box = torch.tensor([[batch_id, class_id, col_min, row_min, col_max, row_max]]).cuda().float()

        final_boxes = torch.cat((final_boxes, local_box), dim=0)
        final_centers = torch.cat((final_centers, object_centers[c, :].unsqueeze(0)), dim=0)

    bounding_boxes = final_boxes[1:, :]
    object_centers = final_centers[1:, :]

    return object_centers, bounding_boxes


def mean_shift_and_loose_bounding_boxes(out_vertex, out_label, extents, xyz_images, depth_mask, intrinsic_matrix):
    batch_size = out_vertex.shape[0]
    stacked_size = out_vertex.shape[1]
    height = out_vertex.shape[2]
    width = out_vertex.shape[3]
    nclasses = stacked_size / 3
    out_label_indices = out_label.unsqueeze(1).long()
    label_onehot = torch.zeros(out_vertex.shape[0], nclasses, out_vertex.shape[2], out_vertex.shape[3]).cuda()
    label_onehot.scatter_(1, out_label_indices, 1)

    extents_batch = extents.repeat(batch_size, 1, 1)
    extents_largest_dim = torch.sqrt((extents_batch * extents_batch).sum(dim=2))
    extents_largest_dim = extents_largest_dim.repeat(1, 3)
    extents_largest_dim = extents_largest_dim.reshape(batch_size, 3, nclasses)
    extents_largest_dim = extents_largest_dim.transpose(1, 2).reshape(batch_size, 3 * nclasses)

    hacky_value = 100.0
    # the point of this is to not merge clusters of different labels
    magnify_mask = torch.ones(label_onehot.shape).cuda().float() * hacky_value
    classes_range = torch.arange(nclasses).cuda().float()
    magnify_mask = magnify_mask * classes_range.unsqueeze(1).unsqueeze(1)

    # b x nclasses x height x width - each channel
    label_class_idx = label_onehot * magnify_mask

    # label: b x nclasses x height x width -> b x nclasses*3 x height x width
    label_onehot_tiled = label_onehot.repeat(1, 1, 3, 1).view(batch_size, -1, height, width)

    mask_repeat = depth_mask.repeat(1, stacked_size, 1, 1)

    label_onehot_tiled = label_onehot_tiled * mask_repeat

    # out_vertex: b x (nclasses * 3) x height x width
    # undoing the scale normalization
    delta_centers = out_vertex.detach() * label_onehot_tiled * extents_largest_dim.unsqueeze(2).unsqueeze(2) * 0.5

    xyz_images = xyz_images.repeat(1, stacked_size / 3, 1, 1)
    xyz_images = xyz_images * label_onehot_tiled
    xyz_images_with_incr = xyz_images - delta_centers
    xyz_images_with_incr = xyz_images_with_incr.reshape(batch_size, nclasses, 3, height, width)

    # batch x nclasses x (3+1) x height x width
    xyz_images_with_incr = torch.cat((xyz_images_with_incr, label_class_idx.unsqueeze(2)), dim=2)

    non_zero_l = torch.nonzero(xyz_images_with_incr[0, :, 3, :, :])

    # concatenate each point with its label, but make this label large so that we don't merge points from different classes
    xs = xyz_images_with_incr[0, non_zero_l[:, 0], 0, non_zero_l[:, 1], non_zero_l[:, 2]].unsqueeze(1)
    ys = xyz_images_with_incr[0, non_zero_l[:, 0], 1, non_zero_l[:, 1], non_zero_l[:, 2]].unsqueeze(1)
    zs = xyz_images_with_incr[0, non_zero_l[:, 0], 2, non_zero_l[:, 1], non_zero_l[:, 2]].unsqueeze(1)
    ls = xyz_images_with_incr[0, non_zero_l[:, 0], 3, non_zero_l[:, 1], non_zero_l[:, 2]].unsqueeze(1)

    xyz_with_increments = torch.cat((xs, ys, zs, ls), dim=1)

    per_cluster_centroids = torch.tensor([]).cuda().float()
    final_boxes = torch.tensor([]).cuda().float()

    if xyz_with_increments.shape[0]:
        cluster_labels = mean_shift_smart_init(xyz_with_increments, dist_threshold=0.05, num_seeds=40, max_iters=30,
                                               batch_size=None)

        # one cluster label for each point
        cluster_labels = cluster_labels.cuda()

        # how many clusters we have
        unique_labels = torch.unique(cluster_labels)

        cluster_labels_onehot = torch.zeros((xyz_with_increments.shape[0], unique_labels.shape[0])).cuda().float()
        cluster_labels_onehot.scatter_(1, cluster_labels.unsqueeze(1).long(), 1)

        per_cluster_points = xyz_with_increments.unsqueeze(1) * cluster_labels_onehot.unsqueeze(2)

        total_points_per_cluster = torch.clamp((cluster_labels_onehot.sum(dim=0)), min=1.0)

        per_cluster_centroids = per_cluster_points.sum(dim=0)
        per_cluster_centroids = per_cluster_centroids / total_points_per_cluster.unsqueeze(1)

        cluster_labels_onehot_long = cluster_labels_onehot.long()
        for k in range(unique_labels.shape[0]):
            onehot = torch.zeros(1, unique_labels.shape[0]).cuda().long()
            onehot[0, k] = 1.0

            point_indices_of_cluster = cluster_labels_onehot_long == onehot
            point_indices_of_cluster = torch.prod(point_indices_of_cluster, 1)
            point_indices_of_cluster = point_indices_of_cluster.unsqueeze(1).repeat(1, 4)

            points_per_cluster = torch.masked_select(xyz_with_increments, point_indices_of_cluster.byte()).reshape(-1, 4)

            per_cluster_centroids[k, :] = torch.median(points_per_cluster, 0)[0]

        percentage_in_each_cluster = total_points_per_cluster * 100.0 / torch.clamp(
            torch.tensor(xyz_with_increments.shape[0]), min=1.0)

        # keep clusters with more than 3% of points. maybe we could do something more intelligent
        valid_clusters = percentage_in_each_cluster > 2

        if not torch.nonzero(valid_clusters).shape[0]:
            per_cluster_centroids = torch.tensor([]).cuda().float()

        else:
            extents_batch = extents.repeat(batch_size, 1, 1)
            extents_largest_dim = torch.sqrt((extents_batch * extents_batch).sum(dim=2))

            final_boxes = torch.ones(1, 6).cuda().float() * (-1)

            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            px = intrinsic_matrix[0, 2]
            py = intrinsic_matrix[1, 2]

            # sort clusters by size. this way, if multi-instance is not enabled, we only use the clusters with the most points
            per_cluster_centroids = torch.cat((per_cluster_centroids, total_points_per_cluster.unsqueeze(1)), dim=1)
            order = torch.sort(per_cluster_centroids[:, 4], descending=True)[1]
            per_cluster_centroids = per_cluster_centroids[order, :]
            valid_clusters = valid_clusters.unsqueeze(1)[order, :]

            per_cluster_centroids_filter = torch.zeros(1, per_cluster_centroids.shape[1]).cuda().float()

            for k in range(per_cluster_centroids.shape[0]):

                class_id = (torch.floor(per_cluster_centroids[k, 3]) / hacky_value).long()

                if not cfg.TEST.MULTI_INSTANCE and class_id in final_boxes[:, 1].long():
                    continue

                if not valid_clusters[k]:
                    continue

                largest_dim = extents_largest_dim[0, class_id]

                z = torch.clamp(per_cluster_centroids[k, 2], min=0.001)
                x_min = per_cluster_centroids[k, 0] - largest_dim / 2.0
                x_max = per_cluster_centroids[k, 0] + largest_dim / 2.0

                y_min = per_cluster_centroids[k, 1] - largest_dim / 2.0
                y_max = per_cluster_centroids[k, 1] + largest_dim / 2.0

                col_min = x_min / z * fx + px
                col_max = x_max / z * fx + px

                row_min = y_min / z * fy + py
                row_max = y_max / z * fy + py

                col_min = torch.clamp(col_min, min=0.0, max=width).int()
                row_min = torch.clamp(row_min, min=0.0, max=height).int()

                col_max = torch.clamp(col_max, min=0.0, max=width).int()
                row_max = torch.clamp(row_max, min=0.0, max=height).int()

                if col_max - col_min <= 2.0 / 100.0 * width or row_max - row_min <= 2.0 / 100.0 * height:
                    continue

                object_segmentation_image = label_onehot[0, class_id, :, :]
                box_area = object_segmentation_image[row_min:row_max, col_min:col_max]
                total_pixels = torch.nonzero(box_area).shape[0]

                coverage = total_pixels * 100.0 / torch.clamp(((row_max - row_min) * (col_max - col_min)), min=1.0)
                if coverage <= 2.0:
                    continue

                # dropped boxes which are too large/small and where segmentation occupies a small portion
                box_local = torch.tensor([[0, class_id, col_min, row_min, col_max, row_max]]).cuda().float()

                final_boxes = torch.cat((final_boxes, box_local), dim=0)
                per_cluster_centroids_filter = torch.cat(
                    (per_cluster_centroids_filter, per_cluster_centroids[k, :].unsqueeze(0)), dim=0)

            final_boxes = final_boxes[1:, :]
            per_cluster_centroids_filter = per_cluster_centroids_filter[1:, :]
            per_cluster_centroids = per_cluster_centroids_filter

    object_centers = per_cluster_centroids
    bounding_boxes = final_boxes

    return object_centers, bounding_boxes
