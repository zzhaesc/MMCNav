import numpy as np
from math import exp, sqrt


def get_metrics(instances, graph):
    tc = 0
    spd = 0
    kpa = 0
    nDTW = 0
    total = 0.0

    all_metrics = list()
    for instance in instances:
        metrics = dict(tc=0.0, spd=0.0, kpa=0.0, nDTW=0.0)

        agent_path = instance['agent_pano_path']
        gold_path = instance['gold_pano_path']
        target_panoid = gold_path[-1]
        total += 1

        def _get_key_points(pano_path):
            if len(pano_path) <= 1:
                return []
            key_points = list((pano_path[0], pano_path[1]))
            for i in range(len(pano_path)):
                pano = pano_path[i]
                if graph.get_num_neighbors(pano) > 2:
                    if i+1 < len(pano_path):
                        next_pano = pano_path[i+1]
                        key_points.append((pano, next_pano))
            return key_points

        gold_key_points = _get_key_points(gold_path)
        agent_key_points = _get_key_points(agent_path)

        kp_correct = len(set(gold_key_points) & set(agent_key_points))

        if agent_path[-1] in graph.get_target_neighbors(gold_path[-1]) + [gold_path[-1]]:
            kp_correct += 1

        _kpa = kp_correct / (len(gold_key_points) + 1)
        metrics['kpa'] = _kpa
        kpa += _kpa

        target_list = graph.get_target_neighbors(target_panoid) + [target_panoid]
        if agent_path[-1] in target_list:
            tc += 1
            metrics['tc'] = 1

        _spd = graph.get_shortest_path_length(agent_path[-1], target_panoid)
        spd += _spd
        metrics['spd'] = _spd

        _nDTW = calculate_nDTW(gold_path, agent_path, graph)
        nDTW += _nDTW
        metrics['nDTW'] = _nDTW

        all_metrics.append(metrics)

    correct = tc
    tc = tc / total * 100
    spd = spd / total
    kpa = kpa / total * 100
    nDTW = nDTW / total * 100
    return correct, tc, spd, kpa, nDTW, all_metrics


def get_metrics_from_results(results, graph, total_token=0):

    instances = list(results['instances'].values())
    correct, tc, spd, kpa, nDTW, all_metrics = get_metrics(instances, graph)

    assert len(instances) == len(all_metrics)
    for instance, metrics in zip(instances, all_metrics):
        instance['metrics'] = metrics

    results['metrics'] = dict(correct=correct, tc=round(tc, 2), spd=round(spd, 2), sed=round(kpa, 2), nDTW=round(nDTW, 2), total_token=total_token)

    return correct, tc, spd, kpa, nDTW, results


def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance between two coordinates (lat, lng).
    """
    lat1, lng1 = coord1['lat'], coord1['lng']
    lat2, lng2 = coord2['lat'], coord2['lng']
    # Convert latitude and longitude from degrees to radians
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    # Haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # Radius of Earth in kilometers (use 6371 for km, 3956 for miles)
    distance = 6371 * c * 10
    return distance

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1['lat'] - coord2['lat']) ** 2 + (coord1['lng'] - coord2['lng']) ** 2) * 10

def manhattan_distance(coord1, coord2):
    return abs(coord1['lat'] - coord2['lat']) + abs(coord1['lng'] - coord2['lng']) * 10


def calculate_dtw_distance(gold_path, agent_path, graph):
    """
    Calculate the DTW distance between two paths.
    """
    len_gold = len(gold_path)
    len_agent = len(agent_path)
    # Initialize the DTW matrix with infinity
    dtw_matrix = np.full((len_gold + 1, len_agent + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Populate the DTW matrix
    for i in range(1, len_gold + 1):
        for j in range(1, len_agent + 1):
            cost = graph.get_shortest_path_length(gold_path[i - 1], agent_path[j - 1]) * 10
            # cost = haversine_distance(gold_path_coords[i - 1], agent_path_coords[j - 1])
            # cost = manhattan_distance(gold_path_coords[i - 1], agent_path_coords[j - 1])
            # Take the minimum of the three neighboring cells plus the cost
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # Insertion
                dtw_matrix[i, j - 1],    # Deletion
                dtw_matrix[i - 1, j - 1] # Match
            )

    # The DTW distance is located at the bottom-right corner of the matrix
    dtw_distance = dtw_matrix[len_gold, len_agent]
    return dtw_distance


def calculate_nDTW(gold_path, agent_path, graph, dth=10.0):
    """
    Calculate the normalized Dynamic Time Warping (nDTW) score between two paths.
    """
    # Extract coordinates for each node in the paths
    # gold_path_coords = [graph.nodes[pano].coordinate for pano in gold_path]
    # agent_path_coords = [graph.nodes[pano].coordinate for pano in agent_path]
    
    # Calculate DTW distance
    dtw_distance = calculate_dtw_distance(gold_path, agent_path, graph)
    
    # Normalize the DTW distance
    normalized_dtw = exp(-dtw_distance / (len(gold_path) * dth))
    return normalized_dtw

