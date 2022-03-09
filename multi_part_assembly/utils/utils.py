import numpy as np


def colorize_part_pc(part_pc, colors):
    """Colorize part point cloud.

    Args:
        part_pc (np.ndarray): [P, N, 3]
        colors (np.ndarray): [max_num_parts, 3 (RGB)]

    Returns:
        np.ndarray: [P, N, 6]
    """
    P, N, _ = part_pc.shape
    colored_pc = np.zeros((P, N, 6))
    colored_pc[:, :, :3] = part_pc
    for i in range(P):
        colored_pc[i, :, 3:] = colors[i]
    return colored_pc
