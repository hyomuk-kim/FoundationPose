import os
from rclpy.node import Node


def get_mesh_file(node: Node) -> str:
    """
    Get the mesh file path from a ROS2 parameter.
    Declares and reads the 'mesh_file' parameter from the given node.
    """
    node.declare_parameter("mesh_file", "")
    mesh_file = node.get_parameter(
        "mesh_file").get_parameter_value().string_value

    if not mesh_file:
        code_dir = os.path.dirname(os.path.realpath(__file__))
        DEFAULT_MESH_FILE = f"{code_dir}/kiri_meshes/cup_ycbv/textured.obj"
        node.get_logger().warn(
            f"No 'mesh_file' parameter provided. Using default: {DEFAULT_MESH_FILE}"
        )
        mesh_file = DEFAULT_MESH_FILE

    assert isinstance(mesh_file,
                      str), f"mesh_file must be a string, got: {mesh_file}"
    assert os.path.exists(mesh_file), f"Mesh file does not exist: {mesh_file}"
    return mesh_file
