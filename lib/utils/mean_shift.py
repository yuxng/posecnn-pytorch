import torch
import numpy as np

def ball_kernel(Z, X, dist):
    """ Computes pairwise ball kernel (without normalizing constant)
        (note this is kernel as defined in non-parametric statistics, not a kernel as in RKHS)

        @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints - the seeds
        @param X: a [m x d] torch.FloatTensor of NORMALIZED datapoints - the points

        @return: a [n x m] torch.FloatTensor of pairwise ball kernel computations,
                 without normalizing constant
    """
    distance = X.unsqueeze(1) - Z.unsqueeze(0)  # a are points, b are seeds
    distance = torch.norm(distance, dim=2)
    within_ball = distance < dist
    within_ball = torch.transpose(within_ball, dim0=0, dim1=1)
    within_ball = within_ball.float()
    return within_ball


def get_label_mode(array):
    """ Computes the mode of elements in an array.
        Ties don't matter. Ties are broken by the smallest value (np.argmax defaults)

        @param array: a numpy array
    """
    labels, counts = np.unique(array, return_counts=True)
    mode = labels[np.argmax(counts)].item()
    return mode


def connected_components(Z, epsilon):
    """
        For the connected components, we simply perform a nearest neighbor search in order:
            for each point, find the points that are up to epsilon away (in cosine distance)
            these points are labeled in the same cluster.

        @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints

        @return: a [n] torch.LongTensor of cluster labels
    """
    n, d = Z.shape

    K = 0
    cluster_labels = torch.ones(n, dtype=torch.long) * -1
    for i in range(n):
        if cluster_labels[i] == -1:

            distances_euclidean = Z.unsqueeze(1) - Z[i:i + 1].unsqueeze(0)  # a are points, b are seeds
            distances_euclidean = torch.norm(distances_euclidean, dim=2)

            component_seeds = distances_euclidean[:, 0] <= epsilon

            # If at least one component already has a label, then use the mode of the label
            if torch.unique(cluster_labels[component_seeds]).shape[0] > 1:
                temp = cluster_labels[component_seeds].numpy()
                temp = temp[temp != -1]
                label = torch.tensor(get_label_mode(temp))
            else:
                label = torch.tensor(K)
                K += 1  # Increment number of clusters

            cluster_labels[component_seeds] = label

    return cluster_labels


def seed_hill_climbing_ball(X, Z, dist_threshold, max_iters=10, batch_size=None):
    """ Runs mean shift hill climbing algorithm on the seeds.
        The seeds climb the distribution given by the KDE of X

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        @param dist_threshold: parameter for the ball kernel
    """
    n, d = X.shape
    m = Z.shape[0]

    for _iter in range(max_iters):

        # Create a new object for Z
        new_Z = Z.clone()

        W = ball_kernel(Z, X, dist_threshold)

        summed_weights = W.sum(dim=1)
        summed_weights = summed_weights.unsqueeze(1)
        summed_weights = torch.clamp(summed_weights, min=1.0)

        # use this allocated weight to compute the new center
        new_Z = torch.mm(W, X)  # Shape: [n x d]

        # Normalize the update
        Z = new_Z / summed_weights

    return Z


def mean_shift_with_seeds(X, Z, dist_threshold, max_iters=10, batch_size=None):
    """ Runs mean-shift

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        @param dist_threshold: parameter for the von Mises-Fisher distribution
    """

    Z = seed_hill_climbing_ball(X, Z, dist_threshold, max_iters=max_iters, batch_size=batch_size)

    # Connected components
    cluster_labels = connected_components(Z, 0.04)  # Set epsilon = 0.1 = 2*alpha

    return cluster_labels, Z


def select_smart_seeds(X, num_seeds, return_selected_indices=False, init_seeds=None, num_init_seeds=None):
    """ Selects seeds that are as far away as possible

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param num_seeds: number of seeds to pick
        @param init_seeds: a [num_seeds x d] vector of initial seeds
        @param num_init_seeds: the number of seeds already chosen.
                               the first num_init_seeds rows of init_seeds have been chosen already

        @return: a [num_seeds x d] matrix of seeds
                 a [n x num_seeds] matrix of distances
    """

    n, d = X.shape

    selected_indices = -1 * torch.ones(num_seeds, dtype=torch.long)

    # Initialize seeds matrix
    if init_seeds is None:
        seeds = torch.empty((num_seeds, d), device=X.device)
        num_chosen_seeds = 0
    else:
        seeds = init_seeds
        num_chosen_seeds = num_init_seeds

    # Keep track of distances
    distances = torch.empty((n, num_seeds), device=X.device)

    if num_chosen_seeds == 0:  # Select first seed if need to
        selected_seed_index = np.random.randint(0, n)
        selected_indices[0] = selected_seed_index
        selected_seed = X[selected_seed_index, :]
        seeds[0, :] = selected_seed

        distances_euclidean = X.unsqueeze(1) - selected_seed.unsqueeze(0)
        distances_euclidean = torch.norm(distances_euclidean, dim=2).squeeze(1)

        distances[:, 0] = distances_euclidean

        num_chosen_seeds += 1
    else:  # Calculate distance to each already chosen seed
        for i in range(num_chosen_seeds):
            ### THIS IS NOT IMPLEMENTED YET ###
            distances[:, i] = .5 * (1 - torch.mm(X, seeds[i:i + 1, :].t())[:, 0])

    # Select rest of seeds
    for i in range(num_chosen_seeds, num_seeds):
        # Find the point that has the furthest distance from the nearest seed
        distance_to_nearest_seed = torch.min(distances[:, :i], dim=1)[0]  # Shape: [n]
        selected_seed_index = torch.argmax(distance_to_nearest_seed)
        selected_indices[i] = selected_seed_index
        selected_seed = torch.index_select(X, 0, selected_seed_index)[0, :]
        seeds[i, :] = selected_seed

        distances_euclidean = X.unsqueeze(1) - selected_seed.unsqueeze(0)
        distances_euclidean = torch.norm(distances_euclidean, dim=2).squeeze(1)

        # Calculate distance to this selected seed
        distances[:, i] = distances_euclidean

    return_tuple = (seeds,)
    if return_selected_indices:
        return_tuple += (selected_indices,)
    return return_tuple


def mean_shift_smart_init(X, dist_threshold, num_seeds=100, max_iters=10, batch_size=None):
    """ Runs mean shift with carefully selected seeds

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param dist_threshold: parameter for the von Mises-Fisher distribution
        @param num_seeds: number of seeds used for mean shift clustering

        @return: a [n] array of cluster labels
    """

    n, d = X.shape

    # Get the seeds
    seeds = select_smart_seeds(X, num_seeds)[0]  # implemented euclidean norm here

    seed_cluster_labels, updated_seeds = mean_shift_with_seeds(X, seeds, dist_threshold, max_iters=max_iters,
                                                               batch_size=batch_size)

    # Get distances to updated seeds
    distances_euclidean = X.unsqueeze(1) - updated_seeds.unsqueeze(0)  # a are points, b are seeds
    distances_euclidean = torch.norm(distances_euclidean, dim=2)
    distances = distances_euclidean

    # Get clusters by assigning point to closest seed
    closest_seed_indices = torch.argmin(distances, dim=1)  # Shape: [n]
    # print closest_seed_indices

    cluster_labels = seed_cluster_labels[closest_seed_indices]

    return cluster_labels
