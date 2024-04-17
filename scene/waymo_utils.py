from typing import NamedTuple
import numpy as np
from PIL import Image
# class CameraInfo(NamedTuple):
#     uid: int
#     R: np.array
#     T: np.array
#     FovY: np.array
#     FovX: np.array
#     image: np.array
#     depth: np.array
#     image_path: str
#     image_name: str
#     width: int
#     height: int
#     timestamp: float = 0.0
#     fl_x: float = -1.0
#     fl_y: float = -1.0
#     cx: float = -1.0
#     cy: float = -1.0
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    sky_mask: np.array = None
    timestamp: float = 0.0
    FovY: float = None
    FovX: float = None
    fl_x: float = None
    fl_y: float = None
    cx: float = None
    cy: float = None
    pointcloud_camera: np.array = None

from utils.graphics_utils import getWorld2View2
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    if 'time' in vertices:
        timestamp = vertices['time'][:, None]
    else:
        timestamp = None
    return BasicPointCloud(points=positions, colors=colors, normals=normals, time=timestamp)


def storePly(path, xyz, rgb, timestamp=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('time', 'f4')]

    normals = np.zeros_like(xyz)
    if timestamp is None:
        timestamp = np.zeros_like(xyz[:, :1])

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, timestamp), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    time_interval: float = 0.02
    time_duration: list = [-0.5, 0.5]


def convertSceneInfo(scene_info_waymo):
    class SceneInfo_new(NamedTuple):
        point_cloud: BasicPointCloud
        train_cameras: list
        test_cameras: list
        nerf_normalization: dict
        ply_path: str

    class CameraInfo_new(NamedTuple):
        uid: int
        R: np.array
        T: np.array
        FovY: np.array
        FovX: np.array
        image: np.array
        depth: np.array
        image_path: str
        image_name: str
        width: int
        height: int
        timestamp: float = 0.0
        fl_x: float = -1.0
        fl_y: float = -1.0
        cx: float = -1.0
        cy: float = -1.0


    train_cameras = []
    test_cameras = []
    for cam_info in scene_info_waymo.train_cameras:
        PIL_image = Image.fromarray((cam_info.image * 255).astype(np.uint8))
        camera_info = CameraInfo_new(uid=cam_info.uid,
                                     R=cam_info.R,
                                     T=cam_info.T,
                                     FovY=cam_info.FovY,
                                     FovX=cam_info.FovX,
                                     image=PIL_image,
                                     image_path=cam_info.image_path,
                                     image_name=cam_info.image_name,
                                     width=cam_info.width,
                                     height=cam_info.height,
                                     timestamp=cam_info.timestamp, depth = None,
                                     fl_x=cam_info.fl_x, fl_y=cam_info.fl_y, cx=cam_info.cx, cy=cam_info.cy)
        train_cameras.append(camera_info)

    for cam_info in scene_info_waymo.test_cameras:
        PIL_image = Image.fromarray((cam_info.image * 255).astype(np.uint8))
        camera_info = CameraInfo_new(uid=cam_info.uid,
                                     R=cam_info.R,
                                     T=cam_info.T,
                                     FovY=cam_info.FovY,
                                     FovX=cam_info.FovX,
                                     image=PIL_image,
                                     image_path=cam_info.image_path,
                                     image_name=cam_info.image_name,
                                     width=cam_info.width,
                                     height=cam_info.height,
                                     timestamp=cam_info.timestamp, depth = None,
                                     fl_x=cam_info.fl_x, fl_y=cam_info.fl_y, cx=cam_info.cx, cy=cam_info.cy)
        test_cameras.append(camera_info)


    converted = SceneInfo_new(point_cloud = scene_info_waymo.point_cloud, train_cameras = train_cameras, test_cameras = test_cameras, nerf_normalization = scene_info_waymo.nerf_normalization, ply_path = scene_info_waymo.ply_path)
    return converted