# vive_ros2/launch/vive_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('vive_ros2')
    rviz_cfg  = os.path.join(pkg_share, 'rviz', 'view_tfs.rviz')

    vive_input = Node(
        package   ='vive_ros2',
        executable='vive_input',
        name      ='vive_input',
        output    ='screen'
    )

    vive_node = Node(
        package   ='vive_ros2',
        executable='vive_node',
        name      ='vive_node',
        arguments =['100'],    
        output    ='screen'
    )

    rviz = Node(
        package   ='rviz2',
        executable='rviz2',
        name      ='rviz2',
        arguments =['-d', rviz_cfg],
        output    ='screen'
    )

    return LaunchDescription([vive_input, vive_node, rviz])
