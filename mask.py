import torch

class SmallTargetFilter:
    def __init__(self, radius=3, sigma_s=1.0, sigma_c=1.0):
        """
        初始化 Small Target Filter (STF) 的参数
        :param radius: 双边滤波的圆形半径
        :param sigma_s: 空间高斯核的标准差
        :param sigma_c: 像素值高斯核的标准差
        """
        self.radius = radius
        self.sigma_s = sigma_s
        self.sigma_c = sigma_c

    def compute_mahalanobis_distance(self, H):
        """
        计算全局马氏距离
        :param H: 高光谱图像 (L, B)
        :return: 马氏距离向量 Z (L,)
        """
        L, B = H.shape
        mean_vector = H.mean(dim=0)  # 均值向量 (B,)
        centered_H = H - mean_vector  # 中心化数据 (L, B)

        # 协方差矩阵及其逆矩阵
        covariance_matrix = torch.cov(centered_H.T)  # (B, B)
        inv_covariance_matrix = torch.linalg.inv(covariance_matrix)  # (B, B)

        # 计算马氏距离
        Z = torch.sum(centered_H @ inv_covariance_matrix * centered_H, dim=1).sqrt()  # (L,)
        return Z

    def bilateral_filter(self, Z, x):
        """
        对马氏距离图 Z 进行双边滤波
        :param Z: 马氏距离图 (M, N)
        :param shape: 图像的空间尺寸 (M, N)
        :return: 滤波后的马氏距离图 ˆZ (M, N)
        """
        M, N = x.shape[0], x.shape[1]
        Z = Z.view(M, N).clone()  # 转换为 2D 形状
        filtered_Z = torch.zeros_like(Z)

        # 提前计算高斯核
        for i in range(M):
            for j in range(N):
                # 提取邻域范围
                x_min, x_max = max(0, i - self.radius), min(M, i + self.radius + 1)
                y_min, y_max = max(0, j - self.radius), min(N, j + self.radius + 1)

                # 提取邻域像素
                window = Z[x_min:x_max, y_min:y_max]
                x, y = torch.meshgrid(
                    torch.arange(x_min, x_max, device=Z.device),
                    torch.arange(y_min, y_max, device=Z.device),
                    indexing='ij'
                )

                # 空间权重
                spatial_weight = torch.exp(-((x - i) ** 2 + (y - j) ** 2) / (2 * self.sigma_s ** 2))

                # 像素值权重
                intensity_weight = torch.exp(-((window - Z[i, j]) ** 2) / (2 * self.sigma_c ** 2))

                # 权重归一化
                weights = spatial_weight * intensity_weight
                weights_sum = weights.sum()

                # 滤波结果
                filtered_Z[i, j] = (window * weights).sum() / weights_sum

        return filtered_Z

    def forward(self, x):
        """
        Small Target Filter 主函数
        :param H: 高光谱图像 (L, B)
        :param shape: 图像的空间尺寸 (M, N)
        :return: 滤波后的掩膜矩阵 ˆZ (M, N)
        """
        H = x.view(-1, x.shape[2])
        # 计算全局马氏距离
        Z = self.compute_mahalanobis_distance(H)  # (L,)

        # 转换为空间矩阵并应用双边滤波
        filtered_Z = self.bilateral_filter(Z, x)  # (M, N)
        return filtered_Z


#
