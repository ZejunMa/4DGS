import os
from tqdm import tqdm
import numpy as np
from PIL import Image

from utils.graphics_utils import BasicPointCloud
from scene.waymo_utils import CameraInfo, getNerfppNorm, fetchPly, storePly, SceneInfo, convertSceneInfo

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.

    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius > 0:
        scale_factor = 1. / fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor


def readWaymoInfo(args):
    cam_infos = []
    car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(args.source_path, "calib"))) if f.endswith('.txt')]
    points = []
    points_time = []

    frame_num = len(car_list)
    if args.frame_interval > 0:
        time_duration = [-args.frame_interval * (frame_num - 1) / 2, args.frame_interval * (frame_num - 1) / 2]
    else:
        time_duration = args.time_duration

    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        ego_pose = np.loadtxt(os.path.join(args.source_path, 'pose', car_id + '.txt'))

        # CAMERA DIRECTION: RIGHT DOWN FORWARDS
        with open(os.path.join(args.source_path, 'calib', car_id + '.txt')) as f:
            calib_data = f.readlines()
            L = [list(map(float, line.split()[1:])) for line in calib_data]
        Ks = np.array(L[:5]).reshape(-1, 3, 4)[:, :, :3]
        lidar2cam = np.array(L[-5:]).reshape(-1, 3, 4)
        lidar2cam = pad_poses(lidar2cam)

        cam2lidar = np.linalg.inv(lidar2cam)
        c2w = ego_pose @ cam2lidar
        w2c = np.linalg.inv(c2w)
        images = []
        image_paths = []
        HWs = []
        for subdir in ['image_0', 'image_1', 'image_2', 'image_3', 'image_4'][:args.cam_num]:
            image_path = os.path.join(args.source_path, subdir, car_id + '.png')
            im_data = Image.open(image_path)
            W, H = im_data.size
            image = np.array(im_data) / 255.
            HWs.append((H, W))
            images.append(image)
            image_paths.append(image_path)

        sky_masks = []
        for subdir in ['sky_0', 'sky_1', 'sky_2', 'sky_3', 'sky_4'][:args.cam_num]:
            sky_data = np.array(Image.open(os.path.join(args.source_path, subdir, car_id + '.png')))
            sky_mask = sky_data > 0
            sky_masks.append(sky_mask.astype(np.float32))

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (len(car_list) - 1)
        point = np.fromfile(os.path.join(args.source_path, "velodyne", car_id + ".bin"),
                            dtype=np.float32, count=-1).reshape(-1, 6)
        point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1)
        point_xyz_world = (np.pad(point_xyz, (0, 1), constant_values=1) @ ego_pose.T)[:, :3]
        points.append(point_xyz_world)
        point_time = np.full_like(point_xyz_world[:, :1], timestamp)
        points_time.append(point_time)
        for j in range(args.cam_num):
            point_camera = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ lidar2cam[j].T)[:, :3]
            R = np.transpose(w2c[j, :3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[j, :3, 3]
            K = Ks[j]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            FovX = FovY = -1.0
            cam_infos.append(CameraInfo(uid=idx * 5 + j, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image=images[j],
                                        image_path=image_paths[j], image_name=car_id,
                                        width=HWs[j][1], height=HWs[j][0], timestamp=timestamp,
                                        pointcloud_camera=point_camera,
                                        fl_x=fx, fl_y=fy, cx=cx, cy=cy,
                                        sky_mask=sky_masks[j]))

        if args.debug_cuda:
            break

    pointcloud = np.concatenate(points, axis=0)
    pointcloud_timestamp = np.concatenate(points_time, axis=0)
    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)

    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    if args.eval:
        # ## for snerf scene
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold == 0]

        # for dynamic scene
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]

        # for emernerf comparison [testhold::testhold]
        if args.testhold == 10:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if
                               (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if
                              (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num) > 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1 / nerf_normalization['radius']

    ply_path = os.path.join(args.source_path, "points3d.ply")
    if not os.path.exists(ply_path):
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0], 3]), normals=None,
                          time=pointcloud_timestamp)
    time_interval = (time_duration[1] - time_duration[0]) / (len(car_list) - 1)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval,
                           time_duration=time_duration)
    # 把waymo的scene_info转换成原版scene_info
    scene_info = convertSceneInfo(scene_info)
    return scene_info
