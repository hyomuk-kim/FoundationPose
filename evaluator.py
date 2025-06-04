import numpy as np
import pyrender
from scipy.spatial.transform import Rotation as R


def compare_masks(mask1, mask2, threshold=0.2):
    # Calculate the intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Handle the case where union is zero to avoid division by zero
    if union == 0:
        if intersection == 0:
            return 1.0, True
        else:
            return 0.0, False

    # Compute Intersection over Union (IoU)
    iou = intersection / union

    # Check if IoU meets the threshold
    return iou, iou >= threshold


def render_depth_and_mask(trimesh_obj, T_C_O, K, image_width=640, image_height=360):
    """
    Render a depth image and mask image given a trimesh object and an object pose.

    Args:
        trimesh_obj: Trimesh object loaded using pyrender.Mesh.from_trimesh().
        T_C_O: 4x4 numpy array representing the object pose relative to the camera.
        K: 3x3 numpy array representing the camera intrinsic matrix.
        image_width: Width of the rendered image (default: 640).
        image_height: Height of the rendered image (default: 360).

    Returns:
        depth_image: 2D numpy array representing the depth image.
        mask: 2D numpy array representing the mask image.
    """
    # Define the rotation matrix for the camera
    R_C2_C = R.from_euler("x", 180, degrees=True).as_matrix()
    T_C2_C = np.eye(4)
    T_C2_C[:3, :3] = R_C2_C
    T_C2_O = T_C2_C @ T_C_O

    # Create a scene
    scene = pyrender.Scene()

    # Add the mesh to the scene
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
    scene.add(mesh, pose=T_C2_O)

    # Set up the camera intrinsics (f_x, f_y, c_x, c_y)
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]

    camera = pyrender.IntrinsicsCamera(fx=f_x, fy=f_y, cx=c_x, cy=c_y)
    scene.add(camera)

    # Create an offscreen renderer
    renderer = pyrender.OffscreenRenderer(image_width, image_height)

    # Render the depth image and compute the mask
    depth_image = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    mask = depth_image > 0

    # Return the depth image and mask
    return depth_image, mask


def render_depth_and_mask_cache(
    trimesh_obj, T_C_O, K, image_width=640, image_height=360
):
    """
    Render a depth image and mask image given a trimesh object and an object pose.

    Args:
        trimesh_obj: Trimesh object loaded using pyrender.Mesh.from_trimesh().
        T_C_O: 4x4 numpy array representing the object pose relative to the camera.
        K: 3x3 numpy array representing the camera intrinsic matrix.
        image_width: Width of the rendered image (default: 640).
        image_height: Height of the rendered image (default: 360).

    Returns:
        depth_image: 2D numpy array representing the depth image.
        mask: 2D numpy array representing the mask image.
    """
    import sys

    def printerr(x):
        print(x, file=sys.stderr)

    if not hasattr(render_depth_and_mask_cache, "first"):
        printerr("~" * 100)
        printerr("FIRST")
        printerr("~" * 100)

        render_depth_and_mask_cache.first = False

        # Define the rotation matrix for the camera
        R_C2_C = R.from_euler("x", 180, degrees=True).as_matrix()
        T_C2_C = np.eye(4)
        T_C2_C[:3, :3] = R_C2_C
        T_C2_O = T_C2_C @ T_C_O

        # Create a scene
        scene = pyrender.Scene()

        # Add the mesh to the scene
        mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
        mesh_node = scene.add(mesh, pose=T_C2_O)

        # Set up the camera intrinsics (f_x, f_y, c_x, c_y)
        f_x = K[0, 0]
        f_y = K[1, 1]
        c_x = K[0, 2]
        c_y = K[1, 2]

        camera = pyrender.IntrinsicsCamera(fx=f_x, fy=f_y, cx=c_x, cy=c_y)
        scene.add(camera)

        # Create an offscreen renderer
        renderer = pyrender.OffscreenRenderer(image_width, image_height)

        # Render the depth image and compute the mask
        depth_image = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        mask = depth_image > 0

        render_depth_and_mask_cache.T_C2_C = T_C2_C
        render_depth_and_mask_cache.scene = scene
        render_depth_and_mask_cache.mesh = mesh
        render_depth_and_mask_cache.mesh_node = mesh_node
        render_depth_and_mask_cache.camera = camera
        render_depth_and_mask_cache.renderer = renderer

        # Return the depth image and mask
        return depth_image, mask
    else:
        printerr("~" * 100)
        printerr("NOT FIRST")
        printerr("~" * 100)
        T_C2_C = render_depth_and_mask_cache.T_C2_C
        mesh_node = render_depth_and_mask_cache.mesh_node
        scene = render_depth_and_mask_cache.scene
        renderer = render_depth_and_mask_cache.renderer

        # Update the pose of the mesh node to move the mesh
        # For example, moving the mesh by applying a new transformation matrix
        T_C2_O = T_C2_C @ T_C_O
        scene.set_pose(mesh_node, pose=T_C2_O)

        # Re-render the scene with the new mesh pose
        depth_image = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        mask = depth_image > 0
        return depth_image, mask
