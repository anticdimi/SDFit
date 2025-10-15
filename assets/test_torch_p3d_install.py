import pytorch3d
import torch
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
)
from pytorch3d.structures import Pointclouds


def test_cuda_support():
    print('\n=== CUDA Support Test ===')
    print(f'PyTorch version: {torch.__version__}')
    print(f'PyTorch3D version: {pytorch3d.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'Number of visible GPUs: {torch.cuda.device_count()}')
        print(f'CUDA device: {torch.cuda.get_device_name()}')
        print(f'CUDA memory allocated: {torch.cuda.memory_allocated(0)}')


def test_point_rasterization():
    print('\n=== Point Rasterization Test ===')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    try:
        # Create point cloud
        points = torch.rand((10, 3), device=device)
        features = torch.ones((10, 3), device=device)
        point_cloud = Pointclouds(points=[points], features=[features])
        print('Point cloud created successfully')

        # Setup and test rasterizer
        raster_settings = PointsRasterizationSettings(
            image_size=128, radius=0.01, points_per_pixel=1
        )
        # Create a simple perspective camera
        R = torch.eye(3, device=device)[None]  # Identity rotation matrix
        T = torch.zeros(1, 3, device=device)  # Zero translation
        cameras = pytorch3d.renderer.PerspectiveCameras(R=R, T=T, device=device)
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(point_cloud)
        print('Rasterization successful')
        print(f'Output shape: {fragments.zbuf.shape}')

    except Exception as e:
        print(f'Error during test: {str(e)}')
        raise


if __name__ == '__main__':
    test_cuda_support()
    test_point_rasterization()
