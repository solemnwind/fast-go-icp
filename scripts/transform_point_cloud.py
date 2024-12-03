from pyntcloud import PyntCloud
import numpy as np
import pandas as pd


def load_point_cloud(file_path):
    """
    Load a point cloud from a PLY file using PyntCloud.
    """
    cloud = PyntCloud.from_file(file_path)
    points = cloud.points[['x', 'y', 'z']].values
    return points, cloud


def normal_distribution_index_sample(points, sample_ratio):
    """
    Sample indices from the point cloud based on a normal distribution.
    """
    num_points = len(points)
    mean_index = num_points // 2
    std_dev = num_points / 100

    indices = np.arange(num_points)
    probabilities = np.exp(-0.5 * ((indices - mean_index) / std_dev) ** 2)
    probabilities /= probabilities.sum()

    num_samples = int(sample_ratio * num_points)
    sampled_indices = np.random.choice(indices, size=num_samples, replace=False, p=probabilities)

    return points[sampled_indices]


def apply_rotation_and_translation(points, rotation_matrix, translation_vector):
    """
    Apply a rotation and translation to a set of points.
    """
    return points @ rotation_matrix.T + translation_vector


def generate_random_rotation_matrix():
    """
    Generate a random 3D rotation matrix.
    """
    angles = np.random.uniform(0, 2 * np.pi, 3)  # 随机角度
    rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


def save_point_cloud(points, original_cloud, file_path):
    """
    Save the point cloud to a file in PLY format using PyntCloud.
    """
    new_cloud = PyntCloud(pd.DataFrame(
        data=points,
        columns=['x', 'y', 'z']
    ))

    if 'red' in original_cloud.points and 'green' in original_cloud.points and 'blue' in original_cloud.points:
        colors = original_cloud.points[['red', 'green', 'blue']].iloc[:len(points)].values
        new_cloud.points['red'], new_cloud.points['green'], new_cloud.points['blue'] = colors.T

    new_cloud.to_file(file_path)


if __name__ == "__main__":
    input_file_path = "data/bunny/Goat skull.ply"
    output_file_path = "data/bunny/Transformed Goat Skull.ply"

    points, original_cloud = load_point_cloud(input_file_path)

    sampled_points = normal_distribution_index_sample(points, sample_ratio=0.1)

    rotation_matrix = generate_random_rotation_matrix()
    translation_vector = np.random.uniform(-5, 5, 3)

    transformed_points = apply_rotation_and_translation(sampled_points, rotation_matrix, translation_vector)

    save_point_cloud(transformed_points, original_cloud, output_file_path)

    print(f"Transformed point cloud saved to {output_file_path}")