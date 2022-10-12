"""Visualize bounding boxes and point clouds"""

import pickle as pkl
import open3d as o3d
import numpy as np
import torch
import scipy
from scipy.spatial import Delaunay

import sys
# sys.path.append('../')
# from lib.utils.kitti_utils import get_objects_from_label


pred_pathname = '/home/alex/gitlab/domain_adaptation_pointrcnn/visual/1142060_output_label.txt'  # .txt, kitti-format
data_pathname = '/home/alex/gitlab/domain_adaptation_pointrcnn/visual/1142060_data.pkl'  # .pkl, contains both box and pc
# pred_pathname = '/Users/lzp/Desktop/pc_sample/1142060_output_label.txt'  # .txt, kitti-format
# data_pathname = '/Users/lzp/Desktop/pc_sample/1142060_data.pkl' 

pc_area_scope = [[-40, 40], [-3,   1], [0, 70.4]]   # x, y, z scope in rect camera coords

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag

def plot3d_old(pts, gt_boxes=None, roi_boxes=None, pred_boxes=None, additional_boxes=None, mask=None, diff_color=False, black_points=False, return_geometry=False):
    '''
        pts: N * 3
        roi_boxes: N * 7
    '''

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if black_points:
        pcd.colors = o3d.utility.Vector3dVector([[0.5,0.5,0.5] for _ in range(len(pts))])

    gt_boxes_lines, roi_boxes_lines, pred_boxes_lines, additional_boxes_lines = [], [], [], []

    if hasattr(mask, 'shape'):
        pts_filtered = pts[mask]
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(pts_filtered)
        pcd_filtered.colors = o3d.utility.Vector3dVector([[0,0,1] for _ in range(len(pts_filtered))])

    if hasattr(gt_boxes, 'shape'):
        for box in gt_boxes:
            # gt_boxes_corners = boxes3d_to_corners3d_rect(*box.tolist())
            gt_boxes_corners = boxes_to_corners_3d(box.reshape(-1, 7))[0]
            gt_boxes_lines.append(corners_to_lines(gt_boxes_corners.tolist(), color=[0,0,0]))

    if hasattr(roi_boxes, 'shape'):
        if diff_color:
            colors = [[1,0,0], [0,1,0], [0,0,1]]
            # colors = [[1,0,0], [1,1,0], [0,0,1]]
            for box, color in zip(roi_boxes, colors):
                roi_boxes_corners = boxes_to_corners_3d(box.reshape(-1, 7))[0]
                roi_boxes_lines.append(corners_to_lines(roi_boxes_corners.tolist(), color=color))
        else:
            for box in roi_boxes:
                roi_boxes_corners = boxes_to_corners_3d(box.reshape(-1, 7))[0]
                roi_boxes_lines.append(corners_to_lines(roi_boxes_corners.tolist(), color=[1,0,0]))
                # roi_boxes_lines =corners_to_lines(roi_boxes_corners.tolist(), color=[1,0,0], thickness=0.02)

    if hasattr(pred_boxes, 'shape'):
        for box in pred_boxes:
            pred_boxes_corners = boxes_to_corners_3d(box.reshape(-1, 7))[0]
            pred_boxes_lines.append(corners_to_lines(pred_boxes_corners.tolist(), color=[0,1,0]))

    if hasattr(additional_boxes, 'shape'):
        for box in additional_boxes:
            additional_boxes_corners = boxes_to_corners_3d(box.reshape(-1, 7))[0]
            additional_boxes_lines.append(corners_to_lines(additional_boxes_corners.tolist(), color=[0,0,1]))
    
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    if hasattr(mask, 'shape'):
        visual = [axis, pcd, pcd_filtered] + gt_boxes_lines + roi_boxes_lines + pred_boxes_lines + additional_boxes_lines
    else:
        visual = [axis, pcd] + gt_boxes_lines + roi_boxes_lines + pred_boxes_lines + additional_boxes_lines
        # visual = [axis, pcd, *roi_boxes_lines]
    if return_geometry:
        return visual
    o3d.visualization.draw_geometries(visual)

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #     R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


# def corners_to_lines(qs, color):
#     """ Draw 3d bounding box in image
#         qs: (8,3) array of vertices for the 3d box in following order:
#         7 -------- 4
#        /|         /|
#       6 -------- 5 .
#       | |        | |
#       . 3 -------- 0
#       |/         |/
#       2 -------- 1

#         Also, show heading by drawing 0-7, 3-4
#     """
#     idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7), (4,1), (5,0)]
#     colors = [color for i in range(14)]

#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(qs),
#         lines=o3d.utility.Vector2iVector(idx),
#     )
#     line_set.colors = o3d.utility.Vector3dVector(colors)

#     return line_set

def corners_to_lines(qs, color, thickness=None):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

        Also, show heading by drawing 0-7, 3-4
    """
    # idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7), (0,7), (3,4)]
    idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7), (4,1), (5,0)]
    
    colors = [color for i in range(14)]

    if thickness is None:
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(qs),
            lines=o3d.utility.Vector2iVector(idx)
        )
        line_set.colors=o3d.utility.Vector3dVector(colors)

    else:
        # use cylinder
        line_mesh = LineMesh(qs, idx, colors, radius=thickness)
        line_set = line_mesh.cylinder_segments

    return line_set

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def boxes3d_to_corners3d_rect(x, y, z, h, w, l, ry):
    """
    :param boxes3d: (N, 7) [x, y, z, l, w, h, ry] in rect coords (KITTI-format)
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
           z (w)
          /
         O--x (l)
         |
         y (h)
    """
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.])
    y_corners = np.array([0, 0, 0, 0, -h,  -h,  -h,  -h])  # center at box bottom
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.])

    corners = np.concatenate((x_corners.reshape(1, 8), y_corners.reshape(1, 8), z_corners.reshape(1, 8)), axis=0)  # (3, 8)

    # counter-clockwise around y
    Rz = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [          0, 1,          0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    rotated_corners = np.matmul(Rz, corners)  # (3, 8)
    rotated_corners += np.array([x, y, z]).reshape(3, 1)

    return rotated_corners.astype(np.float32).T  # (8, 3)


def main():
    # load processed box and point cloud
    with open(data_pathname, 'rb') as f:
        content = pkl.load(f)
    pts_input = content['pts_input']
    gt_boxes3d = content['gt_boxes3d']

    # filter points
    # element-wise multiplication for bitwise and
    # pdb.set_trace()
    flag = (pc_area_scope[0][0] < pts_input[:, 0]) * (pts_input[:, 0] < pc_area_scope[0][1]) * \
           (pc_area_scope[1][0] < pts_input[:, 1]) * (pts_input[:, 1] < pc_area_scope[1][1]) * \
           (pc_area_scope[2][0] < pts_input[:, 2]) * (pts_input[:, 2] < pc_area_scope[2][1])
    pts = pts_input[flag]
    pts_filtered = pts_input[np.logical_not(flag)]

    # render points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(pts_filtered)
    pcd_filtered.colors = o3d.utility.Vector3dVector([[0,0,0] for _ in range(len(pts_filtered))])

    # render gt boxes
    gt_boxes_lines = []
    for box in gt_boxes3d:
        gt_boxes_corners = boxes3d_to_corners3d_rect(*box.tolist())
        gt_boxes_lines.append(corners_to_lines(gt_boxes_corners.tolist(), color=[0,0,0]))

    # render pd boxes
    pd_boxes_lines = []
    for object in get_objects_from_label(pred_pathname):
        corners3d = object.generate_corners3d()
        pd_boxes_lines.append(corners_to_lines(corners3d.tolist(), color=[1,0,0]))

    # visualize
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])

    # visual = [axis, pcd, gt_boxes_lines, pd_boxes_lines]
    visual = [axis, pcd, pcd_filtered] + gt_boxes_lines + pd_boxes_lines
    o3d.visualization.draw_geometries(visual)


if __name__ == '__main__':
    main()
