import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def random_sampling(xyz, nsample):
    """
    隨機取樣函數
    Args:
     xyz: 點雲數據，形狀為 [B, N, 3]
     nsample: 採樣點的數量
    Returns:
     rand_indices: 取樣後的點雲索引，形狀為 [B, nsample]
    """
    B, N, _ = xyz.size()

    # 生成随机索引
    rand_indices = torch.randint(0, N, (B, nsample))

    return rand_indices

def uniform_down_sampling(xyz, nsample, every_k_points=1):
    """
    https://blog.csdn.net/m0_64346597/article/details/128091777
        隨機取樣函數
        Args:
         xyz: 點雲數據，形狀為 [B, N, 3]
         nsample: 採樣點的數量
        Returns:
         indices: 取樣後的點雲索引，形狀為 [B, nsample]
        """
    B, N, _ = xyz.size()

    # 將 xyz 轉換為 numpy.ndarray 類型
    xyz_np = xyz.cpu().numpy() if xyz.is_cuda else xyz.numpy()

    # 創建 open3d 中的 PointCloud 對象並將其點設置為 xyz_o3d
    point_cloud = o3d.geometry.PointCloud()
    for i in range(xyz_np.shape[0]):
        points = o3d.utility.Vector3dVector(xyz_np[i])
        point_cloud.points.extend(points)
    # 使用 uniform_down_sample 函數對點雲進行均勻採樣
    downsampled_point_cloud = point_cloud.uniform_down_sample(every_k_points=every_k_points)

    # 將 downsampled_point_cloud 轉換為 numpy 陣列以獲取點的索引
    downsampled_points_np = np.asarray(downsampled_point_cloud.points)

    # 創建索引張量
    indices = torch.tensor([i for i in range(downsampled_points_np.shape[0])]).unsqueeze(0).repeat(B, 1)
    indices = np.random.choice(indices.shape[0], nsample, replace=False)
    return indices

def vector_angle(x, y):
    Lx = np.sqrt(x.dot(x))
    Ly = (np.sum(y ** 2, axis=1)) ** (0.5)
    cos_angle = np.sum(x * y, axis=1) / (Lx * Ly)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2

def curvature_based_downsampling(xyz, nsample, knn_num=10, angle_thre=70, N=1, C=1):
    """
     曲率下取樣函數
     Args:
      xyz: 點雲數據，形狀為 [B, N, 3]
      nsample: 採樣點的數量
      knn_num: KNN 點的數量，預設為 10
      angle_thre: 角度閾值，預設為 30
      N: 每N個點採樣一次，預設為 5
      C: 取樣均勻性閾值，預設為 10
     Returns:
      indices: 取樣後的點雲索引，形狀為 [B, nsample]
     """
    B, N, _ = xyz.shape
    indices = np.zeros((B, nsample), dtype=int)

    for b in range(B):
        xyz_np = xyz.cpu().numpy() if xyz.is_cuda else xyz.numpy()
        point = xyz_np[b]  # 当前批次的点云数据
        point_size = point.shape[0]
        tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point)))

        # 估计法线
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point))
        o3d.geometry.PointCloud.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))
        normal = np.asarray(pcd.normals)

        # 计算法线角度
        normal_angle = np.zeros(point_size)
        for i in range(point_size):
            _, idx, _ = tree.search_knn_vector_3d(point[i], knn_num + 1)
            current_normal = normal[i]
            knn_normal = normal[idx[1:]]
            normal_angle[i] = np.mean(vector_angle(current_normal, knn_normal))

        # 按照法线角度阈值分割点云
        point_high = point[normal_angle >= angle_thre]
        # point_low = point[normal_angle < angle_thre]

        # 创建高曲率点云对象
        pcd_high = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_high))

        # 创建低曲率点云对象
        # pcd_low = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_low))

        # 对高曲率点云进行均匀采样
        # pcd_high_down = pcd_high.uniform_down_sample(every_k_points=N)
        #
        # # 对低曲率点云进行均匀采样
        # pcd_low_down = pcd_low.uniform_down_sample(every_k_points=C)

        # 合并高曲率和低曲率采样后的点云
        # merged_points = np.concatenate((np.asarray(pcd_high_down.points), np.asarray(pcd_low_down.points)))
        merged_points = np.asarray(pcd_high.points)
        # 从合并的点云中随机选择 nsample 个点
        if len(pcd_high.points) > nsample:
            # 將點和其對應的曲率值進行排序
            selected_indices = np.random.choice(len(pcd_high.points), nsample, replace=False)
            # 将选取的点的索引保存到结果中
            indices[b] = selected_indices

        if len(pcd_high.points) < nsample:
            num_to_add = nsample - merged_points.shape[0]
            additional_points = point[:num_to_add]
            merged_points = np.concatenate((merged_points, additional_points))
            # 将合并点云的索引保存到结果中
            indices[b] = np.arange(merged_points.shape[0])

    return indices