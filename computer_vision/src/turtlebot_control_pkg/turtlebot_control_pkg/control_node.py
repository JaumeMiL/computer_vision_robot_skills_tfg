#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int8
from sensor_msgs.msg import LaserScan


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        self.declare_parameter('gesture_topic', 'gesture_command')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')  # millor absolut

        gesture_topic = self.get_parameter('gesture_topic').get_parameter_value().string_value
        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        self.sub_gest = self.create_subscription(Int8, gesture_topic, self.gest_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data)

        # Forcem TwistStamped perquè és el que t’està funcionant
        self.pub_vel = self.create_publisher(TwistStamped, cmd_vel_topic, 10)

        # IDs (VisionNode)
        self.STOP_IDS = {0, 2, 8}
        self.ID_REPEAT = 1

        self.ID_BACKWARD = 3
        self.ID_TURN_LEFT = 4
        self.ID_FORWARD = 5
        self.ID_FOLLOW_RIGHT = 6
        self.ID_FOLLOW_LEFT = 7
        self.ID_ROTATE = 9
        self.ID_TURN_RIGHT = 10

        # Velocitats
        self.lin_speed = 0.2     # <-- el que vols
        self.ang_speed = 0.8

        self.t_rotate = 8.0
        self.t_fwd = 5.0
        self.t_bwd = 5.0
        self.t_turn_90 = 2.0

        # Follow (làser)
        self.wall_presence_dist = 0.50
        self.follow_forward_lin = 0.20
        self.follow_turn_ang = 0.50
        self.follow_recover_ang = 0.30
        self.follow_timeout_s = 30.0

        self.front_sector_deg = 20.0
        self.obst_front_dist = 0.35

        self.accept_after_t = 0.0
        self.post_action_cooldown = 0.6

        self.last_executed_id = None
        self.rearmed = True
        self.last_action_id = None

        self.last_scan = None

        # Timed action
        self.action_active = False
        self.action_start_t = 0.0
        self.action_duration = 0.0
        self.action_lin = 0.0
        self.action_ang = 0.0

        # Follow FSM
        self.mode_follow = None
        self.follow_state = None
        self.preferred_turn = "RIGHT"
        self.mode_follow_deadline_t = 0.0

        self.follow_log_period = 0.5
        self._last_follow_log_t = 0.0

        # Repeat one-shot latch
        self.repeat_latched = False
        self.last_repeat_msg_t = 0.0
        self.repeat_release_gap_s = 0.7

        self.create_timer(0.05, self.loop)

        self.get_logger().info("CONTROL NODE LLEST.")
        self.get_logger().info(f"Topics: gesture='{gesture_topic}', scan='{scan_topic}', cmd_vel='{cmd_vel_topic}' (TwistStamped=True)")

    def scan_callback(self, msg: LaserScan):
        self.last_scan = msg

    def _min_range_in_sector(self, scan: LaserScan, center_deg: float, half_width_deg: float) -> float:
        if scan is None:
            return float('inf')

        center = math.radians(center_deg)
        half = math.radians(half_width_deg)

        amin = scan.angle_min
        ainc = scan.angle_increment
        n = len(scan.ranges)

        rmin = float('inf')
        for i in range(n):
            ang = amin + i * ainc
            if (center - half) <= ang <= (center + half):
                d = scan.ranges[i]
                if math.isfinite(d) and d > scan.range_min:
                    rmin = min(rmin, d)
        return rmin

    def _front_distance(self) -> float:
        return self._min_range_in_sector(self.last_scan, center_deg=0.0, half_width_deg=self.front_sector_deg)

    def _obstacle_ahead(self) -> bool:
        return self._front_distance() < self.obst_front_dist

    def _side_distance(self, side: str) -> float:
        if side == "right":
            return self._min_range_in_sector(self.last_scan, center_deg=-90.0, half_width_deg=15.0)
        return self._min_range_in_sector(self.last_scan, center_deg=90.0, half_width_deg=15.0)

    def _wall_present(self, side: str) -> bool:
        d = self._side_distance(side)
        return math.isfinite(d) and (d < self.wall_presence_dist)

    def _busy(self) -> bool:
        return self.action_active or (self.mode_follow is not None)

    def _arm_again(self):
        self.rearmed = True

    def _latch_execute(self, act_id: int):
        self.last_executed_id = act_id
        self.rearmed = False

    def _stop_follow(self):
        self.mode_follow = None
        self.follow_state = None
        self.mode_follow_deadline_t = 0.0
        self.preferred_turn = "RIGHT"

    def _cancel_timed_action(self):
        self.action_active = False
        self.action_lin = 0.0
        self.action_ang = 0.0
        self.action_duration = 0.0

    def _cancel_all(self, reason: str):
        self.get_logger().info(f"CANCEL: {reason}")
        self._stop_follow()
        self._cancel_timed_action()
        self.accept_after_t = time.monotonic() + self.post_action_cooldown
        self.send_vel(0.0, 0.0)

    def _start_timed_action(self, lin: float, ang: float, duration: float, name: str):
        self._stop_follow()
        self.action_active = True
        self.action_start_t = time.monotonic()
        self.action_duration = float(duration)
        self.action_lin = float(lin)
        self.action_ang = float(ang)
        self.get_logger().info(f"START {name}: lin={lin:.2f} ang={ang:.2f} dur={duration:.1f}s")

    def _start_follow(self, side: str):
        self._cancel_timed_action()
        self.mode_follow = side
        self.follow_state = "IDLE"
        self.mode_follow_deadline_t = time.monotonic() + self.follow_timeout_s
        self.preferred_turn = "RIGHT"
        self.get_logger().info(f"FOLLOW {side.upper()} START (30s max)")

    def _repeat_last(self):
        if self.last_action_id is None:
            self.get_logger().warn("REPEAT: no hi ha last_action")
            return
        act = self.last_action_id
        self.get_logger().info(f"REPEAT last_action={act}")
        self.rearmed = True
        self._handle_action(act, from_repeat=True)

    def gest_callback(self, msg: Int8):
        act = int(msg.data)
        now = time.monotonic()

        if act in self.STOP_IDS:
            self._cancel_all(f"EMERGENCY STOP (id={act})")
            self._arm_again()
            return

        if act == self.ID_REPEAT:
            self.last_repeat_msg_t = now
            if (not self.repeat_latched) and (not self._busy()) and (now >= self.accept_after_t):
                self.repeat_latched = True
                self._repeat_last()
            return

        if now < self.accept_after_t:
            return

        if self._busy():
            return

        if (self.last_executed_id is not None) and (act == self.last_executed_id) and (not self.rearmed):
            return

        self._handle_action(act, from_repeat=False)

    def _handle_action(self, act: int, from_repeat: bool = False):
        if act == self.ID_FOLLOW_RIGHT:
            self._latch_execute(act)
            self._start_follow("right")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_FOLLOW_LEFT:
            self._latch_execute(act)
            self._start_follow("left")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_ROTATE:
            self._latch_execute(act)
            self._start_timed_action(0.0, +self.ang_speed, self.t_rotate, "ROTATE")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_FORWARD:
            self._latch_execute(act)
            self._start_timed_action(+self.lin_speed, 0.0, self.t_fwd, "FORWARD")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_BACKWARD:
            self._latch_execute(act)
            self._start_timed_action(-self.lin_speed, 0.0, self.t_bwd, "BACKWARD")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_TURN_LEFT:
            self._latch_execute(act)
            self._start_timed_action(0.0, -self.ang_speed, self.t_turn_90, "TURN_LEFT")
            if not from_repeat:
                self.last_action_id = act
            return

        if act == self.ID_TURN_RIGHT:
            self._latch_execute(act)
            self._start_timed_action(0.0, +self.ang_speed, self.t_turn_90, "TURN_RIGHT")
            if not from_repeat:
                self.last_action_id = act
            return

        self.get_logger().warn(f"Acció desconeguda {act} (ignorada)")

    def loop(self):
        now = time.monotonic()

        if self.repeat_latched and (now - self.last_repeat_msg_t) > self.repeat_release_gap_s:
            self.repeat_latched = False

        if self.mode_follow in ("right", "left"):
            if now >= self.mode_follow_deadline_t:
                self.get_logger().info("FOLLOW TIMEOUT -> STOP")
                self._stop_follow()
                self.accept_after_t = time.monotonic() + self.post_action_cooldown
                self.send_vel(0.0, 0.0)
                return

            self._follow_fsm_step(self.mode_follow)
            self._log_follow(now, self.mode_follow)
            return

        if self.action_active:
            elapsed = now - self.action_start_t
            if elapsed >= self.action_duration:
                self._cancel_timed_action()
                self.accept_after_t = time.monotonic() + self.post_action_cooldown
                self.send_vel(0.0, 0.0)
                return

            self.send_vel(self.action_lin, self.action_ang)
            return

        self.send_vel(0.0, 0.0)

    def _follow_fsm_step(self, side: str):
        obstacle = self._obstacle_ahead()
        wall_left = self._wall_present("left")
        wall_right = self._wall_present("right")

        follow_side_present = wall_right if side == "right" else wall_left

        if self.follow_state == "IDLE":
            if obstacle:
                self.follow_state = "DECIDE"
                return
            self.send_vel(self.follow_forward_lin, 0.0)
            return

        if self.follow_state == "DECIDE":
            if (not wall_right) and wall_left:
                self.preferred_turn = "RIGHT"
            elif (not wall_left) and wall_right:
                self.preferred_turn = "LEFT"
            else:
                self.preferred_turn = "RIGHT"
            self.follow_state = "TURNING"
            return

        if self.follow_state == "TURNING":
            if not obstacle:
                self.follow_state = "FOLLOW"
                self.send_vel(self.follow_forward_lin, 0.0)
                return
            ang = -self.follow_turn_ang if self.preferred_turn == "RIGHT" else +self.follow_turn_ang
            self.send_vel(0.0, ang)
            return

        if self.follow_state == "FOLLOW":
            if obstacle:
                self.follow_state = "TURNING"
                return
            if not follow_side_present:
                self.follow_state = "RECOVER"
                return
            self.send_vel(self.follow_forward_lin, 0.0)
            return

        if self.follow_state == "RECOVER":
            if obstacle:
                self.follow_state = "TURNING"
                return
            if follow_side_present:
                self.follow_state = "FOLLOW"
                self.send_vel(self.follow_forward_lin, 0.0)
                return

            ang = -self.follow_recover_ang if side == "right" else +self.follow_recover_ang
            self.send_vel(self.follow_forward_lin, ang)

    def _log_follow(self, now: float, side: str):
        if now - self._last_follow_log_t < self.follow_log_period:
            return
        self._last_follow_log_t = now

        front = self._front_distance()
        dl = self._side_distance("left")
        dr = self._side_distance("right")
        wl = self._wall_present("left")
        wr = self._wall_present("right")
        remaining = max(0.0, self.mode_follow_deadline_t - now)

        self.get_logger().info(
            f"FOLLOW[{side}] st={self.follow_state} front={front:.2f} "
            f"L={dl:.2f}({int(wl)}) R={dr:.2f}({int(wr)}) t_left={remaining:.1f}"
        )

    def send_vel(self, lin: float, ang: float):
        # Mateix “estil” que el ros2 topic pub: TwistStamped amb header neutre
        m = TwistStamped()
        m.header.stamp.sec = 0
        m.header.stamp.nanosec = 0
        m.header.frame_id = ''
        m.twist.linear.x = float(lin)
        m.twist.angular.z = float(ang)
        self.pub_vel.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = TwistStamped()
        stop.header.stamp.sec = 0
        stop.header.stamp.nanosec = 0
        node.pub_vel.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


