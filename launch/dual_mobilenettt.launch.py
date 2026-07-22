from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument('x', default_value='-10.9'),
        DeclareLaunchArgument('y', default_value='-4.0'),
        DeclareLaunchArgument('z', default_value='0'),
        DeclareLaunchArgument('yaw', default_value='0'),
        DeclareLaunchArgument('use_sim', default_value='true', 
                              description='Use simulation (true) or real hardware (false)'),
        DeclareLaunchArgument('left_camera_device', default_value='/dev/video4',
                              description='Left camera device path for real hardware'),
        DeclareLaunchArgument('right_camera_device', default_value='/dev/video5',
                              description='Right camera device path for real hardware'),
        DeclareLaunchArgument('robot_namespace', default_value='j100_0681',
                              description='Robot namespace for real hardware'),
        DeclareLaunchArgument('start_delay', default_value='10.0',
                              description='Delay in seconds before controller starts'),
    ]
    
    use_sim = LaunchConfiguration('use_sim')
    left_camera_device = LaunchConfiguration('left_camera_device')
    right_camera_device = LaunchConfiguration('right_camera_device')
    robot_namespace = LaunchConfiguration('robot_namespace')
    start_delay = LaunchConfiguration('start_delay')
    
    pkg_share = FindPackageShare('mobilenettt_visionnavigation')
    xacro_path = PathJoinSubstitution([pkg_share, 'urdf', 'jackal_gazebo.urdf.xacro'])
    world_path = PathJoinSubstitution([pkg_share, 'GazeboWorlds', 'corridor_2.world'])
    
    robot_description = ParameterValue(
        Command(['xacro ', xacro_path]),
        value_type=str
    )
    
    set_gz_plugin_env = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib'
    )
    
    # ============ SIMULATION ONLY ============
    gz_world = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path],
        output='screen',
        condition=IfCondition(use_sim)
    )
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}, {'use_sim_time': use_sim}],
        output='screen',
        condition=IfCondition(use_sim)
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
        output='screen',
        condition=IfCondition(use_sim)
    )
    
    # Simulation controller spawners
    spawner_jsb_sim = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster'],
        output='screen',
        condition=IfCondition(use_sim)
    )
    
    spawner_diffdrive_sim = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'jackal_velocity_controller'],
        output='screen',
        condition=IfCondition(use_sim)
    )
    
    load_jsb_sim = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[spawner_jsb_sim],
        ),
        condition=IfCondition(use_sim)
    )
    
    load_diffdrive_sim = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawner_jsb_sim,
            on_exit=[spawner_diffdrive_sim],
        ),
        condition=IfCondition(use_sim)
    )
    
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/jackal_velocity_controller/cmd_vel@geometry_msgs/msg/TwistStamped@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/left_camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/left_camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/right_camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/right_camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
        ],
        output='screen',
        condition=IfCondition(use_sim)
    )
    
    # ============ REAL HARDWARE ONLY ============
    left_usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='left_usb_cam',
        parameters=[{
            'video_device': left_camera_device,
            'image_width': 640,
            'image_height': 480,
            'framerate': 30.0,
            'pixel_format': 'yuyv',
            'brightness': 0,
            'contrast': 50,
            'saturation': 64,
            'gain': 64,
            'auto_exposure': False,
            'exposure': 166,
        }],
        remappings=[
            ('image_raw', '/left_camera/image'),
        ],
        output='screen',
        condition=UnlessCondition(use_sim)
    )
    
    right_usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='right_usb_cam',
        parameters=[{
            'video_device': right_camera_device,
            'image_width': 640,
            'image_height': 480,
            'framerate': 30.0,
            'pixel_format': 'yuyv',
            'brightness': 0,
            'contrast': 50,
            'saturation': 64,
            'gain': 64,
            'auto_exposure': False,
            'exposure': 166,
        }],
        remappings=[
            ('image_raw', '/right_camera/image'),
        ],
        output='screen',
        condition=UnlessCondition(use_sim)
    )
    
    load_jsb_real = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '-c', '/j100_0681/controller_manager', 'joint_state_broadcaster'],
        output='screen',
        condition=UnlessCondition(use_sim)
    )
    
    load_velocity_real = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '-c', '/j100_0681/controller_manager', 'platform_velocity_controller'],
        output='screen',
        condition=UnlessCondition(use_sim)
    )
    
    activate_jsb_real = ExecuteProcess(
        cmd=['ros2', 'control', 'set_controller_state', '-c', '/j100_0681/controller_manager', 'joint_state_broadcaster', 'active'],
        output='screen',
        condition=UnlessCondition(use_sim)
    )
    
    activate_velocity_real = ExecuteProcess(
        cmd=['ros2', 'control', 'set_controller_state', '-c', '/j100_0681/controller_manager', 'platform_velocity_controller', 'active'],
        output='screen',
        condition=UnlessCondition(use_sim)
    )
    
    chain_load_velocity = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_jsb_real,
            on_exit=[load_velocity_real],
        ),
        condition=UnlessCondition(use_sim)
    )
    
    chain_activate_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_velocity_real,
            on_exit=[activate_jsb_real],
        ),
        condition=UnlessCondition(use_sim)
    )
    
    chain_activate_velocity = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=activate_jsb_real,
            on_exit=[activate_velocity_real],
        ),
        condition=UnlessCondition(use_sim)
    )
    
    # ============ COMMON NODES ============
    # ONE node for TTT computation (handles both cameras)
    ttt_dual_node = Node(
        package='mobilenettt_visionnavigation',
        executable='dual_compute_ttt.py',
        name='tttvalues_dual',
        parameters=[{'use_sim_time': use_sim}],
        output='screen'
    )
    
    # ONE controller node for simulation
    controller_node_sim = Node(
        package='mobilenettt_visionnavigation',
        executable='dual_controller_sidefacing.py',  # ← FIXED: Was dual_compute_ttt.py
        name='controller_dual',
        parameters=[
            {'left_camera_topic': '/left_camera/image'},
            {'right_camera_topic': '/right_camera/image'},
            {'left_ttt_topic': '/left_ttt_values'},
            {'right_ttt_topic': '/right_ttt_values'},
            {'velocity_topic': '/jackal_velocity_controller/cmd_vel'},
            {'use_sim_time': True}
        ],
        output='screen',
        condition=IfCondition(use_sim)
    )
    
    # ONE controller node for real hardware
    controller_node_real = Node(
        package='mobilenettt_visionnavigation',
        executable='dual_controller_sidefacing.py',
        name='controller_dual',
        parameters=[
            {'left_camera_topic': '/left_camera/image'},
            {'right_camera_topic': '/right_camera/image'},
            {'left_ttt_topic': '/left_ttt_values'},
            {'right_ttt_topic': '/right_ttt_values'},
            {'velocity_topic': '/j100_0681/platform/cmd_vel'},
            {'use_sim_time': False}
        ],
        output='screen',
        condition=UnlessCondition(use_sim)
    )
    
    delayed_controller_sim = TimerAction(
        period=start_delay,
        actions=[controller_node_sim],
    )
    
    delayed_controller_real = TimerAction(
        period=start_delay,
        actions=[controller_node_real],
    )
    
    return LaunchDescription(declared_args + [
        set_gz_plugin_env,
        gz_world,
        robot_state_publisher,
        spawn_entity,
        load_jsb_sim,
        load_diffdrive_sim,
        bridge,
        left_usb_cam_node,
        right_usb_cam_node,
        load_jsb_real,
        chain_load_velocity,
        chain_activate_jsb,
        chain_activate_velocity,
        # Only ONE TTT node
        ttt_dual_node,
        # Only ONE controller (delayed)
        delayed_controller_sim,
        delayed_controller_real,
    ])
