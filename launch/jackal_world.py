from launch import LaunchDescription
from launch.actions import ExecuteProcess  # Add this import at the top if not already present
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Set up paths
    vbn_dir = get_package_share_directory('mobilenettt_visionnavigation')

    # Launch arguments
    front_flea3_arg = DeclareLaunchArgument(
        'front_flea3', default_value='false',
        description='Use Flea3 camera configuration'
    )

    default_config = LaunchConfiguration('default_config')
    front_flea3 = LaunchConfiguration('front_flea3')

    base_config_arg = DeclareLaunchArgument(
        'default_config', default_value='base',
        condition=UnlessCondition(front_flea3)
    )

    flea_config_arg = DeclareLaunchArgument(
        'default_config', default_value='front_flea3',
        condition=IfCondition(front_flea3)
    )

    config_arg = DeclareLaunchArgument(
        'config', default_value=default_config,
        description='Jackal configuration'
    )

    # Include Gazebo world
# Launch Gazebo with your world using gz sim
    gz_sim_world = ExecuteProcess(
        cmd=['gz', 'sim', os.path.join(vbn_dir, 'GazeboWorlds', 'corridor.world')],
        output='screen'
    )


    # Include spawn Jackal launch
    spawn_jackal_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vbn_dir, 'launch', 'spawn_jackal.py')
        ),
        launch_arguments={
            'x': '0',
            'y': '0',
            'z': '1.0',
            'yaw': '0',
            'config': LaunchConfiguration('config')
        }.items()
    )

    return LaunchDescription([
        front_flea3_arg,
        base_config_arg,
        flea_config_arg,
        config_arg,
        gz_sim_world,
        spawn_jackal_launch
    ])
