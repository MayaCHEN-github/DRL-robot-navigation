import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray



GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        # 环境参数初始化
        self.environment_dim = environment_dim 
        self.odom_x = 0 # 机器人当前X坐标
        self.odom_y = 0 # 机器人当前Y坐标
        
        # 添加标志来跟踪里程计数据是否准备好
        self.odom_ready = False

        self.goal_x = 1 # 目标X坐标
        self.goal_y = 0.0 # 目标Y坐标

        self.upper = 5.0 # 目标X坐标的生成范围
        self.lower = -5.0 # 目标Y坐标的生成范围
        self.velodyne_data = np.ones(self.environment_dim) * 10 # 初始化激光雷达数据全为10（表示10m距离，远距离表示无障碍物）
        self.last_odom = None # 存储最新的里程计数据

        # 机器人状态（名称、位置、姿态）初始化
        self.set_self_state = ModelState() 
        self.set_self_state.model_name = "r1" # 机器人模型名称
        # 机器人位置（X, Y, Z）和姿态（X, Y, Z, W）初始化
        self.set_self_state.pose.position.x = 0.0 # 机器人当前X坐标
        self.set_self_state.pose.position.y = 0.0 # 机器人当前Y坐标
        self.set_self_state.pose.position.z = 0.0 # 机器人当前Z坐标
        self.set_self_state.pose.orientation.x = 0.0 # 机器人当前X轴旋转
        self.set_self_state.pose.orientation.y = 0.0 # 机器人当前Y轴旋转
        self.set_self_state.pose.orientation.z = 0.0 # 机器人当前Z轴旋转
        self.set_self_state.pose.orientation.w = 1.0 # 机器人当前旋转四元数

        # 将360度激光雷达数据分割成environment_dim个角度区间
        """
        激光雷达原始数据是3D点云数据，而强化学习算法需要固定维度的输入。
        每个区间代表一个方向上的障碍物距离，将360度激光雷达数据分割成environment_dim个角度区间
        经过处理后，self.velodyne_data 变成一个长度为 environment_dim 的数组：每个元素代表一个角度区间的障碍物距离，距离值越小，表示障碍物越近。
        距离值为10表示该方向无障碍物。
        """
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim] # 上一个区间的结束角度,下一个区间的结束角度
            )
        self.gaps[-1][-1] += 0.03  # 最后一个区间稍微扩展

        port = "11311"  # ROS默认端口
        subprocess.Popen(["roscore", "-p", port])  # 启动ROS核心

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)  # 初始化ROS节点
        # 构建launch文件路径
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")
        # 启动Gazebo仿真
        # 启动Gazebo仿真，添加--gui=false参数禁用图形界面
        # 启动Gazebo仿真，添加--gui=false参数禁用图形界面
        # 注意：--gui参数需要放在launch文件路径之前
        subprocess.Popen(["roslaunch", "-p", port, "--gui=false", fullpath])
        print("Gazebo launched with GUI disabled!")

        # Set up the ROS publishers and subscribers
        # 创建ROS发布者（Publisher）
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1) # 发布机器人速度指令
        # 创建ROS发布者（Publisher）
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        # 创建ROS服务代理（ServiceProxy）
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty) # 恢复Gazebo物理仿真
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty) # 暂停Gazebo物理仿真
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty) # 重置Gazebo世界
        # 创建ROS发布者（Publisher）
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3) # 目标点
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1) # 线速度
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1) # 角速度
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1 # 订阅激光雷达数据（雷达数据从这里传入！）
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1 # 订阅里程计数据
        )
        
        # 等待里程计数据初始化
        print("等待里程计数据初始化...")
        timeout = 10.0  # 10秒超时
        start_time = rospy.Time.now()
        while not self.odom_ready and (rospy.Time.now() - start_time).to_sec() < timeout:
            rospy.sleep(0.1)
        
        if self.odom_ready:
            print("里程计数据初始化成功！")
        else:
            print("警告：里程计数据初始化超时，可能影响训练效果")

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v): # 激光雷达数据的接收函数
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10 # 初始化激光雷达数据全为10（表示10m距离，远距离表示无障碍物）
        for i in range(len(data)):
            if data[i][2] > -0.2: # 只考虑地面以上的点
                # 计算点与机器人之间的夹角
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                # 计算点与机器人之间的距离
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                # 找到对应的角度区间，更新最小距离
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data): # 从里程计数据中获取机器人的最新位置信息
        self.last_odom = od_data
        self.odom_ready = True  # 标记里程计数据已准备好

    # Perform an action and read a new state 执行智能体的动作并返回环境反馈。
    def step(self, action):
        target = False

        # 等待里程计数据准备好
        if not self.odom_ready:
            # 如果里程计数据还没有准备好，等待一下
            rospy.sleep(0.1)
            if not self.odom_ready:
                # 如果仍然没有准备好，抛出异常或重置环境
                print("Error: Odometry data not ready after waiting. Resetting environment...")
                # 尝试重置环境来获取新的里程计数据
                self.reset()
                # 如果重置后仍然没有数据，抛出异常
                if not self.odom_ready:
                    raise RuntimeError("Failed to get odometry data after reset. Check ROS topics and Gazebo simulation.")

        # Publish the robot action 发布机器人的动作指令
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0] # 线速度
        vel_cmd.angular.z = action[1] # 角速度
        self.vel_pub.publish(vel_cmd) # 发布到ROS函数
        self.publish_markers(action) # 在RViz中显示可视化标记

        # 步进式仿真，通过暂停/恢复控制仿真的时间步长，避免连续仿真消耗过多计算资源。
        rospy.wait_for_service("/gazebo/unpause_physics") # 等待服务可用（恢复Gazebo物理仿真）
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds 让仿真运行TIME_DELTA秒（0.1秒）
        time.sleep(TIME_DELTA)

        # 暂停物理仿真
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state 读取激光雷达数据
        done, collision, min_laser = self.observe_collision(self.velodyne_data) # 检测碰撞
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data 从里程计数据获取机器人位置
        # 检查里程计数据是否可用，如果不可用则使用默认值
        if self.last_odom is None:
            # 如果里程计数据还没有准备好，使用默认值
            self.odom_x = 0.0
            self.odom_y = 0.0
            angle = 0.0
        else:
            self.odom_x = self.last_odom.pose.pose.position.x
            self.odom_y = self.last_odom.pose.pose.position.y
            quaternion = Quaternion(
                self.last_odom.pose.pose.orientation.w,
                self.last_odom.pose.pose.orientation.x,
                self.last_odom.pose.pose.orientation.y,
                self.last_odom.pose.pose.orientation.z,
            )
            euler = quaternion.to_euler(degrees=False)
            angle = round(euler[2], 4) # 获取Z轴旋转角度（偏航角）

        # Calculate distance to the goal from the robot 计算机器人与目标点的欧几里得距离
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal 计算机器人与目标点的相对角度，告诉机器人应该朝哪个方向转向才能面向目标
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward 检测是否到达目标点（如果距离小于GOAL_REACHED_DIST，则认为到达目标点）
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]] # 构建完整的机器人状态（距离、角度、线速度、角速度）
        state = np.append(laser_state, robot_state) # 将激光雷达数据和机器人状态合并，形成最终的状态向量
        reward = self.get_reward(target, collision, action, min_laser) # 计算奖励
        return state, reward, done, target # 返回状态、奖励、是否完成、是否到达目标点

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # 在reset方法中，我们需要重新计算角度，因为angle变量在这里没有定义
        # 从机器人的当前位置和姿态计算角度
        if self.last_odom is not None:
            quaternion = Quaternion(
                self.last_odom.pose.pose.orientation.w,
                self.last_odom.pose.pose.orientation.x,
                self.last_odom.pose.pose.orientation.y,
                self.last_odom.pose.pose.orientation.z,
            )
            euler = quaternion.to_euler(degrees=False)
            angle = round(euler[2], 4)
        else:
            # 如果里程计数据不可用，使用默认角度
            angle = 0.0

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        # 随机生成目标点，并检查是否在障碍物上。
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        # 随机重新布置环境中的障碍物，避免过拟合&提升鲁棒性
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz 在RViz（ROS的可视化工具）中发布可视化标记。
        # 创建三种不同的可视化标记: 目标点位置（绿色圆柱体），机器人线速度（红色立方体）、机器人角速度（红色长方体）。
        # 在RViz中会看到：
        # 绿色圆柱体：显示目标点位置
        # 红色立方体1：在右侧显示线速度（宽度变化）
        # 红色立方体2：在右侧稍上方显示角速度（宽度变化）
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        """从激光雷达数据中检测机器人是否即将发生碰撞。如果任何方向的障碍物距离 < 0.35米，认为即将碰撞，返回碰撞状态和最小距离。"""
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        """根据当前状态计算奖励值."""
        if target:
            return 100.0 # 如果机器人到达目标点，奖励100分
        elif collision:
            return -100.0 # 如果机器人即将发生碰撞，奖励-100分
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0 # 距离奖励函数
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2 
            # action[0] / 2：鼓励前进（线速度越大，奖励越高）。
            # abs(action[1]) / 2：鼓励转向（角速度越大，奖励越高）。
            # r3(min_laser) / 2：鼓励远离障碍物（障碍物距离越小，奖励越高）。
