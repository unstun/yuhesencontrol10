#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import os
import sys
import threading
import matplotlib.pyplot as plt
from nav_msgs.msg import Path
import tf2_ros
# 注意：本脚本使用 map -> base_link 的 TF 变换作为机器人实际位置

class LivePlotter:
    def __init__(self):
        # 1. 初始化节点
        rospy.init_node('live_trajectory_plotter', anonymous=True)
        
        # 2. 参数配置
        self.plan_topic = "/move_base_rmp/PathPlanner/plan" # 规划路径话题
        self.target_frame = "2d_map"       # 地图坐标系
        self.source_frame = "base_footprint" # 机器人基座坐标系 (如果你的机器人叫 body 或 base_footprint，请修改这里)
        self.save_dir = os.path.expanduser("~/trajectory_data")
        
        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 3. 数据存储 (使用线程锁保证多线程安全)
        self.lock = threading.Lock()
        self.actual_x = []
        self.actual_y = []
        self.planned_x = []
        self.planned_y = []
        
        # 4. TF 监听器初始化 (获取定位的核心)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 5. 绘图初始化
        plt.ion() # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("Trajectory Monitor: TF Location vs Global Plan")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.grid(True)
        self.ax.axis('equal')

        # 初始化曲线对象
        # 实际轨迹：蓝色实线
        self.line_actual, = self.ax.plot([], [], 'b-', label='Actual (TF)', linewidth=1.5)
        # 规划路径：绿色虚线
        self.line_planned, = self.ax.plot([], [], 'g--', label='Planned (Global)', linewidth=2, alpha=0.8)
        self.ax.legend(loc='upper right')

        # 6. 订阅规划话题
        rospy.Subscriber(self.plan_topic, Path, self.plan_callback)

        rospy.loginfo("========================================")
        rospy.loginfo(f"绘图节点已就绪！")
        rospy.loginfo(f"实际位置来源: TF 查询 ({self.target_frame} <-> {self.source_frame})")
        rospy.loginfo(f"规划路径来源: {self.plan_topic}")
        rospy.loginfo("数据将在 Ctrl+C 退出时保存至 ~/trajectory_data")
        rospy.loginfo("========================================")

        # 注册关闭时的回调
        rospy.on_shutdown(self.save_data)

    def get_current_location(self):
        """通过 TF 获取当前机器人在 map 下的坐标"""
        try:
            # 查询最近时刻的变换 (timeout设为0.1秒)
            trans = self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, rospy.Time(0), rospy.Duration(0.1))
            return trans.transform.translation.x, trans.transform.translation.y
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            # 如果暂时获取不到 TF (比如刚启动)，返回 None
            return None, None

    def plan_callback(self, msg):
        """
        规划回调：包含【防止路径消失】的逻辑
        只有当目标改变，或路径变得更长时才更新，防止机器人走动时路径缩短。
        """
        # 1. 过滤空路径
        if not msg.poses:
            return

        # 提取新路径的点
        new_x = []
        new_y = []
        for pose in msg.poses:
            new_x.append(pose.pose.position.x)
            new_y.append(pose.pose.position.y)
        
        with self.lock:
            # 情况A: 第一次收到路径，直接存入
            if not self.planned_x:
                self.planned_x = new_x
                self.planned_y = new_y
                return

            # 情况B: 判断是否是同一个目标
            # 计算【新路径终点】和【已存路径终点】的距离
            dx = new_x[-1] - self.planned_x[-1]
            dy = new_y[-1] - self.planned_y[-1]
            dist_end = (dx**2 + dy**2)**0.5

            # 阈值设为 1.0 米 (如果终点变动超过1米，认为是新目标)
            if dist_end > 1.0:
                rospy.loginfo("检测到新的导航目标，更新规划路径...")
                self.planned_x = new_x
                self.planned_y = new_y
            else:
                # 情况C: 终点相同（同一个任务中）
                # 仅当新路径比旧路径【更长】时才更新 (通常是刚开始规划时)
                # 这样当机器人走动导致路径变短时，我们依然保留最长的那条完整路径
                if len(new_x) > len(self.planned_x):
                    self.planned_x = new_x
                    self.planned_y = new_y
                # 否则：保持原来的完整路径不动

    def update_plot(self):
        """主循环调用：获取定位并更新图表"""
        # 1. 主动获取当前定位
        cur_x, cur_y = self.get_current_location()
        
        with self.lock:
            # 只有成功获取到坐标才记录
            if cur_x is not None:
                self.actual_x.append(cur_x)
                self.actual_y.append(cur_y)
            
            # 复制数据用于绘图 (避免多线程读写冲突)
            ax_data = list(self.actual_x)
            ay_data = list(self.actual_y)
            px_data = list(self.planned_x)
            py_data = list(self.planned_y)

        # 如果完全没有数据，跳过
        if not ax_data and not px_data:
            return

        # 2. 更新线条
        if ax_data:
            self.line_actual.set_data(ax_data, ay_data)
        if px_data:
            self.line_planned.set_data(px_data, py_data)

        # 3. 动态调整视野 (自适应缩放)
        all_x = ax_data + px_data
        all_y = ay_data + py_data
        
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            margin = 1.0 # 视野边距
            self.ax.set_xlim(min_x - margin, max_x + margin)
            self.ax.set_ylim(min_y - margin, max_y + margin)

        # 4. 刷新画布
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        """循环执行"""
        rate = rospy.Rate(10) # 10Hz 刷新率
        while not rospy.is_shutdown():
            try:
                self.update_plot()
                rate.sleep()
            except Exception as e:
                rospy.logwarn(f"绘图循环出错: {e}")
                break

    def save_data(self):
        """保存 CSV 和 图片"""
        print("\n正在保存最终数据...")
        csv_path = os.path.join(self.save_dir, 'trajectory_log.csv')
        img_path = os.path.join(self.save_dir, 'final_result.png')
        
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Type', 'X', 'Y'])
                with self.lock:
                    for x, y in zip(self.actual_x, self.actual_y):
                        writer.writerow(['Actual', x, y])
                    for x, y in zip(self.planned_x, self.planned_y):
                        writer.writerow(['Planned', x, y])
            print(f"数据已保存至: {csv_path}")
            
            self.fig.savefig(img_path)
            print(f"对比图已保存至: {img_path}")
        except Exception as e:
            print(f"保存失败: {e}")

if __name__ == '__main__':
    try:
        plotter = LivePlotter()
        plotter.run()
    except rospy.ROSInterruptException:
        pass