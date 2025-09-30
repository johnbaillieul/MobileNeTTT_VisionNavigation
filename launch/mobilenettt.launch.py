from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument('x', default_value='-10'),
        DeclareLaunchArgument('y', default_value='-4.5'),
        DeclareLaunchArgument('z', default_value='0'),
        DeclareLaunchArgument('yaw', default_value='0'),
    ]

    pkg_share = FindPackageShare('mobilenettt_visionnavigation')
    xacro_path = PathJoinSubstitution([pkg_share, 'urdf', 'jackal_gazebo.urdf.xacro'])
    #yaml_file = PathJoinSubstitution([pkg_share, 'config', 'control.yaml'])
    world_path = PathJoinSubstitution([pkg_share, 'GazeboWorlds', 'corridor_2.world'])

    robot_description = ParameterValue(
        Command(['xacro ', xacro_path]),
        value_type=str
    )

    set_gz_plugin_env = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib'
    )

    gz_world = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}, {'use_sim_time': True}],
        output='screen'
    )

    spawn_entity = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_sim', 'create',
            '--name', 'jackal',
            '--x', LaunchConfiguration('x'),
            '--y', LaunchConfiguration('y'),
            '--z', LaunchConfiguration('z'),
            '--Y', LaunchConfiguration('yaw'),
            '--topic', 'robot_description'
        ],
        output='screen'
    )
    
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",

        ],        
        output='screen'
    )

    
    ttt_node = Node(
        package='mobilenettt_visionnavigation',
        executable='compute_ttt.py',
        name='tttvalues',
        parameters=[{'image_sub_name': '/camera/image'}, {'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription(declared_args + [
        set_gz_plugin_env,
        gz_world,
        robot_state_publisher,
        spawn_entity,
        bridge,
        ttt_node
    ])
