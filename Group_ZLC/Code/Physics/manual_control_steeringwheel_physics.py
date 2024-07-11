#!/usr/bin/env python
# 用中文解释代码
# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
# from map4_create_scene import *
import pandas as pd

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

import winsound
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

player_points = [[64.5671043395996, -67.8489532470703, 180], [-51.9316291809082, 40.5525245666504, 90],[30.5017871856689,141.272964477539,0]]
barrer_points = [[-24, -67.8489532470703, 90], [-51.9316291809082, 109.501, 180],[100.543891906738,119.686325073242,90]]
max_count = 0
success_flag = True
# 定义一个函数find_weather_presets，用于查找天气预设
def find_weather_presets():
    # 编译一个正则表达式，用于匹配单词
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    # 定义一个函数name，用于将匹配到的单词用空格连接
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    # 获取carla.WeatherParameters中的所有属性，并匹配以大写字母开头的属性
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    # 返回一个列表，其中包含carla.WeatherParameters中的属性值和name函数处理后的属性名
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

# 定义一个函数get_actor_display_name，用于获取 actor 的显示名称
def get_actor_display_name(actor, truncate=250):
    # 将actor.type_id中的下划线替换为点，并将点后的字符串首字母大写，其他字符小写
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    # 如果字符串长度超过truncate，则将超出部分用省略号表示
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
n = 0
class World(object):
    def __init__(self, carla_world, hud, actor_filter):
        # 初始化世界类
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        # 获取天气预设
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        # 每帧调用hud的on_world_tick方法
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            global n
            n = random.randint(0, 2)
            # n = 2
            if n == 0:
                spawn_point.location.x = player_points[0][0]
                spawn_point.location.y = player_points[0][1]
                spawn_point.rotation.yaw = player_points[0][2]
            elif n == 1:
                spawn_point.location.x = player_points[1][0]
                spawn_point.location.y = player_points[1][1]
                spawn_point.rotation.yaw = player_points[1][2]
            elif n == 2:
                spawn_point.location.x = player_points[2][0]
                spawn_point.location.y = player_points[2][1]
                spawn_point.rotation.yaw = player_points[2][2]
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # print(spawn_point.rotation.yaw)
            # print('----------------')
            # print(spawn_point.location.x)
            # print(spawn_point.location.y)
            # print(spawn_point.location.z)
            # print('----------------')
            # transform = carla.Transform(Location(x=47.5, y=-57.2, z=0.6), Rotation(yaw=180))
            # transform = carla.Transform(Location(x=98.1, y=93.9, z=5.1), Rotation(yaw=180))
            # self.player = self.world.try_spawn_actor(blueprint, transform)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        # 切换天气，reverse为True时表示反向切换
        self._weather_index += -1 if reverse else 1
        # 将_weather_index取模于_weather_presets的长度，使其一直在0-23之间切换
        self._weather_index %= len(self._weather_presets)
        # 获取切换后的天气配置
        preset = self._weather_presets[self._weather_index]
        # 显示天气信息
        self.hud.notification('Weather: %s' % preset[1])
        # 设置天气
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        # 调用hud的tick方法，将当前的clock传入
        self.hud.tick(self, clock)

    def render(self, display):
        # 调用相机管理器的渲染方法，将相机拍摄的画面渲染到指定的显示器上
        self.camera_manager.render(display)
        # 调用HUD的渲染方法，将车辆信息渲染到指定的显示器上
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================

# 定义DualControl类，用于实现手动控制和自动驾驶的切换
# class DualControl(object):
#     def __init__(self, world, start_in_autopilot):
#         # 初始化DualControl类，world为CarlaWorld类实例，start_in_autopilot表示是否开启自动驾驶
#         self._autopilot_enabled = start_in_autopilot
#         # 判断world.player的类型，如果是carla.Vehicle，则设置车辆控制，并设置自动驾驶状态
#         if isinstance(world.player, carla.Vehicle):
#             self._control = carla.VehicleControl()
#             world.player.set_autopilot(self._autopilot_enabled)
#         # 如果是carla.Walker，则设置行人类控制，并将自动驾驶状态设置为False
#         elif isinstance(world.player, carla.Walker):
#             self._control = carla.WalkerControl()
#             self._autopilot_enabled = False
#             # 获取行人的旋转角度
#             self._rotation = world.player.get_transform().rotation
#         else:
#             # 如果不是这两种类型，则抛出异常
#             raise NotImplementedError("Actor type not supported")
#         # 设置方向盘的缓存值为0
#         self._steer_cache = 0.0
#         # 在hud上显示提示信息
#         world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
#
#         # initialize steering wheel
#         pygame.joystick.init()
#
#         # 获取连接的joystick数量
#         joystick_count = pygame.joystick.get_count()
#         # 如果连接的joystick数量大于1，抛出异常
#         if joystick_count > 1:
#             raise ValueError("Please Connect Just One Joystick")
#
#         # 获取第一个joystick
#         # self._joystick = pygame.joystick.Joystick(0)
#         # 初始化joystick
#         self._joystick.init()
#
#         # 读取配置文件
#         self._parser = ConfigParser()
#         self._parser.read('wheel_config.ini')
#         # 获取配置文件中的参数
#         self._steer_idx = int(
#             self._parser.get('G29 Racing Wheel', 'steering_wheel'))
#         self._throttle_idx = int(
#             self._parser.get('G29 Racing Wheel', 'throttle'))
#         self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
#         self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
#         self._handbrake_idx = int(
#             self._parser.get('G29 Racing Wheel', 'handbrake'))
#
#
#     def parse_events(self, world, clock):
#         # 遍历所有的事件
#         for event in pygame.event.get():
#             # 如果事件类型是退出
#             if event.type == pygame.QUIT:
#                 # 返回True
#                 return True
#             # 如果事件类型是游戏手柄按钮按下
#             elif event.type == pygame.JOYBUTTONDOWN:
#                 # 如果按钮是0
#                 if event.button == 0:
#                     # 重新开始游戏
#                     world.restart()
#                 # 如果按钮是1
#                 elif event.button == 1:
#                     # 切换信息
#                     world.hud.toggle_info()
#                 # 如果按钮是2
#                 elif event.button == 2:
#                     # 切换摄像头
#                     world.camera_manager.toggle_camera()
#                 # 如果按钮是3
#                 elif event.button == 3:
#                     # 切换天气
#                     world.next_weather()
#                 # 如果按钮是倒车档
#                 elif event.button == self._reverse_idx:
#                     # 如果当前是正挡，则切换为倒挡，否则切换为正挡
#                     self._control.gear = 1 if self._control.reverse else -1
#                 # 如果按钮是23
#                 elif event.button == 23:
#                     # 切换传感器
#                     world.camera_manager.next_sensor()
#
#             elif event.type == pygame.KEYUP:
#                 # 检查是否按下退出快捷键
#                 if self._is_quit_shortcut(event.key):
#                     return True
#                 # 按下退格键，重新开始
#                 elif event.key == K_BACKSPACE:
#                     world.restart()
#                 # 按下F1键，切换显示信息
#                 elif event.key == K_F1:
#                     world.hud.toggle_info()
#                 # 按下h或/键，切换帮助
#                 elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
#                     world.hud.help.toggle()
#                 # 按下Tab键，切换摄像头
#                 elif event.key == K_TAB:
#                     world.camera_manager.toggle_camera()
#                 # 按下Shift+C，切换天气
#                 elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
#                     world.next_weather(reverse=True)
#                 # 按下C，切换天气
#                 elif event.key == K_c:
#                     world.next_weather()
#                 # 按下`键，切换传感器
#                 elif event.key == K_BACKQUOTE:
#                     world.camera_manager.next_sensor()
#                 # 按下1-9键，切换传感器
#                 elif event.key > K_0 and event.key <= K_9:
#                     world.camera_manager.set_sensor(event.key - 1 - K_0)
#                 # 按下R键，切换录制
#                 elif event.key == K_r:
#                     world.camera_manager.toggle_recording()
#                 # 如果是车辆控制，按下Q键，切换 gear
#                 if isinstance(self._control, carla.VehicleControl):
#
#                     if event.key == K_q:
#                         self._control.gear = 1 if self._control.reverse else -1
#                     # 按下M键，切换手动/自动 gear
#                     elif event.key == K_m:
#                         self._control.manual_gear_shift = not self._control.manual_gear_shift
#                         self._control.gear = world.player.get_control().gear
#                         world.hud.notification('%s Transmission' %
#                                                ('Manual' if self._control.manual_gear_shift else 'Automatic'))
#                     # 按下，.键，减档
#                     elif self._control.manual_gear_shift and event.key == K_COMMA:
#                         self._control.gear = max(-1, self._control.gear - 1)
#                     # 按下。键，加档
#                     elif self._control.manual_gear_shift and event.key == K_PERIOD:
#                         self._control.gear = self._control.gear + 1
#                     # 按下P键，切换自动/手动驾驶
#                     elif event.key == K_p:
#                         global begin_flag
#                         begin_flag = True
#
#                         self._autopilot_enabled = not self._autopilot_enabled
#                         world.player.set_autopilot(self._autopilot_enabled)
#                         world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
#
#         # 如果自动驾驶功能没有开启
#         if not self._autopilot_enabled:
#             # 如果控制对象是车辆
#             if isinstance(self._control, carla.VehicleControl):
#                 # 解析车辆按键
#                 self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
#                 # 解析车辆方向盘
#                 self._parse_vehicle_wheel(world)
#                 # 控制车辆倒车
#                 self._control.reverse = self._control.gear < 0
#             # 如果控制对象是行人
#             elif isinstance(self._control, carla.WalkerControl):
#                 # 解析行人按键
#                 self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
#             # 将控制指令应用到车辆
#             world.player.apply_control(self._control)
#
#     def _parse_vehicle_keys(self, keys, milliseconds):
#         # 判断是否按下 K_UP 或 w 键，若按下，油门值设为 1.0，否则设为 0.0
#         self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
#         # 设置方向盘转角增量
#         steer_increment = 5e-4 * milliseconds
#         # 判断是否按下 K_LEFT 或 a 键，若按下，方向盘转角减去增量
#         if keys[K_LEFT] or keys[K_a]:
#             self._steer_cache -= steer_increment
#         # 判断是否按下 K_RIGHT 或 d 键，若按下，方向盘转角加上增量
#         elif keys[K_RIGHT] or keys[K_d]:
#             self._steer_cache += steer_increment
#         # 若没有按下方向键，方向盘转角设为 0.0
#         else:
#             self._steer_cache = 0.0
#         # 限制方向盘转角在 -0.7 到 0.7 之间
#         self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
#         # 设置方向盘转角值
#         self._control.steer = round(self._steer_cache, 1)
#         # 判断是否按下 K_DOWN 或 s 键，若按下，刹车值设为 1.0，否则设为 0.0
#         self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
#         # 判断是否按下 K_SPACE 键，若按下，手刹值设为 True，否则设为 False
#         self._control.hand_brake = keys[K_SPACE]
#
#     def _parse_vehicle_wheel(self,world):
#         # 获取操纵杆的数量
#         numAxes = self._joystick.get_numaxes()
#         # 获取操纵杆的输入
#         jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
#         # print (jsInputs)
#         # 获取操纵杆的按钮
#         jsButtons = [float(self._joystick.get_button(i)) for i in
#                      range(self._joystick.get_numbuttons())]
#
#         # 将输入范围[1，-1]映射到输出[0,1]的自定义函数，即输入的1表示没有按下任何内容
#         # For the steering, it seems fine as it is
#         K1 = 1.0  # 0.55
#         steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])
#
#         K2 = 1.6  # 1.6
#         throttleCmd = K2 + (2.05 * math.log10(
#             -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
#         if throttleCmd <= 0:
#             throttleCmd = 0
#         elif throttleCmd > 1:
#             throttleCmd = 1
#
#         brakeCmd = 1.6 + (2.05 * math.log10(
#             -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
#         if brakeCmd <= 0:
#             brakeCmd = 0
#         elif brakeCmd > 1:
#             brakeCmd = 1
#
#         # 限速
#         # 获取当前车辆
#         vehicle = world.player
#         # 设置一个目标速度（单位：m/s）
#         target_speed = 32.0  # 例如，10 m/s
#         # 获取当前速度
#         current_velocity = vehicle.get_velocity()
#         current_speed = 3.6 * math.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2 + current_velocity.z ** 2)
#         # 计算需要的油门和刹车值（这里只是一个简单的示例，实际情况可能需要更复杂的逻辑）
#         # throttle = 0.0
#         if current_speed > target_speed:
#             # brake = min((current_speed - target_speed) / 10.0, 1.0)  # 逐渐增加刹车直到减速到目标速度
#             brakeCmd = 1.0
#
#         self._control.steer = steerCmd
#         self._control.brake = brakeCmd
#         self._control.throttle = throttleCmd
#
#         #toggle = jsButtons[self._reverse_idx]
#
#         self._control.hand_brake = bool(jsButtons[self._handbrake_idx])
#
#     def _parse_walker_keys(self, keys, milliseconds):
#         # 设置速度为0
#         self._control.speed = 0.0
#         # 如果按下向下或者s键，设置速度为0
#         if keys[K_DOWN] or keys[K_s]:
#             self._control.speed = 0.0
#         # 如果按下向左或者a键，设置速度为0.01，并且旋转角度减少0.08*milliseconds
#         if keys[K_LEFT] or keys[K_a]:
#             self._control.speed = .01
#             self._rotation.yaw -= 0.08 * milliseconds
#         # 如果按下向右或者d键，设置速度为0.01，并且旋转角度增加0.08*milliseconds
#         if keys[K_RIGHT] or keys[K_d]:
#             self._control.speed = .01
#             self._rotation.yaw += 0.08 * milliseconds
#         # 如果按下向上或者w键，设置速度为5.556，如果按下shift键，则速度为2.778
#         if keys[K_UP] or keys[K_w]:
#             self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
#         # 设置跳跃键为空格键
#         self._control.jump = keys[K_SPACE]
#         # 将旋转角度四舍五入到小数点后一位
#         self._rotation.yaw = round(self._rotation.yaw, 1)
#         # 设置方向为旋转角度的前向向量
#         self._control.direction = self._rotation.get_forward_vector()
#
#     @staticmethod
#     def _is_quit_shortcut(key):
#         # 检查按键是否为退出快捷键
#         return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == pygame.locals.K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == pygame.locals.K_v:
                    world.next_map_layer()
                elif event.key == pygame.locals.K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == pygame.locals.K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == pygame.locals.K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == pygame.locals.K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == pygame.locals.K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == pygame.locals.K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == pygame.locals.K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == pygame.locals.K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == pygame.locals.K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == pygame.locals.K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == pygame.locals.K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == pygame.locals.K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == pygame.locals.K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == pygame.locals.K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

# 时间，位置，速度
mytime = []
myx = []
myy = []
myv = []
TOR_cnt_list = []
isTOR_list = []
info_take = 0
TOR_cnt = 0
begin_flag = False
'''
TOR_cnt = 0
isTOR = False
timeCnt = 0
timeMargin = 0.5
......
timeCnt += deltaTime
if timeCnt > timeMargin:
    if not isTOR:
        if mode == "random":
            ......
        elif mode == "agent":
            ......
    timeCnt -= timeMargin
    TOR_cnt += 1
    dump(TOR_cnt, isTOR)    
    if isTOR:
        alert
        
'''
# 生成障碍物
def get_accident_scene_0(client):

    scene_list = []
    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    obj = random.choice(world.get_blueprint_library().filter(
        'vehicle.toyota.prius'))  # world.get_blueprint_library().find('walker.pedestrian.0003') ########直接反手生成个障碍物
    obj.set_attribute('role_name', 'd')
    # spawn_point = spawn_points[11]
    spawn_point = random.choice(spawn_points)
    # generate barrier
    # spawn_point.location.x = 64.5671043395996
    # spawn_point.location.y = -67.8489532470703
    transform = carla.Transform(carla.Location(x=barrer_points[0][0], y=barrer_points[0][1], z=0.6), carla.Rotation(yaw=barrer_points[0][2]))
    vehicle_barrier = world.try_spawn_actor(obj, transform)
    vehicle_barrier.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
    scene_list.append([vehicle_barrier])
    transform = carla.Transform(carla.Location(x=barrer_points[1][0], y=barrer_points[1][1], z=0.6), carla.Rotation(yaw=barrer_points[1][2]))
    vehicle_barrier = world.try_spawn_actor(obj, transform)
    vehicle_barrier.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
    scene_list.append([vehicle_barrier])
    transform = carla.Transform(carla.Location(x=barrer_points[2][0], y=barrer_points[2][1], z=0.6),
                                carla.Rotation(yaw=barrer_points[2][2]))
    vehicle_barrier = world.try_spawn_actor(obj, transform)
    vehicle_barrier.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
    scene_list.append([vehicle_barrier])
    # transform = carla.Transform(carla.Location(x=17.1, y=-67.8489532470703, z=0.6), carla.Rotation(yaw=-180))
    # vehicle_barrier = world.try_spawn_actor(obj, transform)
    # vehicle_barrier.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
    # scene_list.append([vehicle_barrier])

    # print(spawn_point.location.x)
    # print(spawn_point.location.y)
    # print(spawn_point.location.z)
    # print('*******')
    return scene_list

# 判断车辆前方 vehicle控制车辆
def is_vehicle_in_front(vehicle, target_vehicle, distance_threshold=10.0):
    # 获取两辆车的transform
    # vehicle_transform = vehicle.get_transform()
    target_transform = target_vehicle.get_transform()
    vehicle_transform = vehicle
    # target_transform = target_vehicle
    # 计算两车之间的向量
    relative_vector = target_transform.location - vehicle_transform.location

    # 获取主车辆的朝向向量（这里简化处理，只考虑Z轴旋转，即yaw）
    # 注意：CARLA中transform的rotation是四元数，这里简化为仅使用yaw
    # 在实际应用中，你可能需要从四元数中提取完整的朝向信息
    vehicle_forward = carla.Vector3D(x=math.cos(vehicle_transform.rotation.yaw),
                                      y=math.sin(vehicle_transform.rotation.yaw), z=0.0)


    # 计算两向量之间的点积，并转换为角度差（这里简化处理，仅作为示例）
    # 注意：这种方法可能不够精确，因为它没有考虑完整的3D朝向
    dot_product = np.dot(np.array([relative_vector.x, relative_vector.y]),
                         np.array([vehicle_forward.x, vehicle_forward.y]))
    try:
        angle_difference = math.acos(dot_product / (relative_vector.length() * vehicle_forward.length())) * (180.0 / math.pi)

        # 检查角度差是否在可接受的范围内（这里假设是正前方的一个小角度）
        if angle_difference < 5.0 and relative_vector.length() < distance_threshold:
            return True
        else:
            return False
    finally:

        return False


class HUD(object):
    def __init__(self, width, height):
        # 初始化HUD类，设置HUD的宽度和高度
        self.dim = (width, height)
        # 初始化字体
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        # 设置字体名称
        font_name = 'courier' if os.name == 'nt' else 'mono'
        # 获取所有字体
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        # 设置默认字体
        default_font = 'ubuntumono'
        # 如果默认字体在字体列表中，则使用默认字体，否则使用第一个字体
        mono = default_font if default_font in fonts else fonts[0]
        # 匹配字体
        mono = pygame.font.match_font(mono)
        # 初始化mono字体
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        # 初始化FadingText类
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        # 初始化HelpText类
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        # 初始化服务器帧率
        self.server_fps = 0
        # 初始化帧数
        self.frame = 0
        # 初始化模拟时间
        self.simulation_time = 0
        # 设置显示信息为True
        self._show_info = True
        # 初始化信息文本
        self._info_text = []
        # 初始化服务器时钟
        self._server_clock = pygame.time.Clock()
        self.firsttime = 0.0


    def on_world_tick(self, timestamp):
        # 更新服务器时钟
        self._server_clock.tick()
        # 获取服务器帧率
        self.server_fps = self._server_clock.get_fps()
        # 获取当前帧数
        self.frame = timestamp.frame
        # 获取模拟时间
        self.simulation_time = timestamp.elapsed_seconds
    # 每一帧更新
    def tick(self, world, clock):
        # 更新通知
        self._notifications.tick(world, clock)
        # 如果不显示信息，则直接返回
        if not self._show_info:
            return
        # 获取玩家车辆的变换、速度和控制信息
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()

        # 计算方向
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        # 获取碰撞历史
        colhist = world.collision_sensor.get_collision_history()
        # 计算碰撞强度
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        # 获取所有车辆
        vehicles = world.world.get_actors().filter('vehicle.*')

        '''
        这段代码是用于在Carla模拟器中显示车辆和行人的信息。Carla是一个开源的自动驾驶汽车模拟器，用于开发和测试自动驾驶汽车技术。
        这段代码主要用于在模拟器中显示车辆和行人的实时信息，如速度、方向、位置等。
        具体来说，这段代码的主要功能如下：
        1.获取服务器帧率、客户端帧率、车辆名称、地图名称、模拟时间等信息，并将其格式化为字符串。
        2.计算车辆的速度、方向、位置等信息，并将其格式化为字符串。
        3.如果控制对象是车辆，则显示车辆的油门、转向、刹车等信息。
        4.如果控制对象是行人，则显示行人的速度、跳跃等信息。
        5.检查车辆是否发生碰撞，并显示碰撞信息。
        6.显示车辆数量和附近车辆的信息。

        需要注意的是，这段代码只是一个示例，实际应用中可能需要根据具体需求进行修改和优化。
        '''
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        # 获取时间，位置x, y,速度
        global info_take
        global TOR_cnt
        global begin_flag
        if begin_flag and (self.simulation_time) - self.firsttime >= 0.25:
            # mytime.append(datetime.timedelta(seconds=int(self.simulation_time)))
            mytime.append(self.simulation_time)
            myx.append(t.location.x)
            myy.append(t.location.y)
            myv.append(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

            if info_take == 2:
                isTOR_list.append(1)
            else:
                isTOR_list.append(0)

            TOR_cnt_list.append(TOR_cnt)
            self.firsttime = self.simulation_time


        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            # print(t.rotation.yaw)

            vehicless = vehicles
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            # if info_take == 0:
            #     for x in vehicless:
            #         if x.id != world.player.id:
            #             if is_vehicle_in_front(t, x, 40):
            #
            #                 # self.notification("please takeover")
            #                 info_take = 1
                            # auto_flag = not auto_flag  # True 是自动驾驶
            for d, vehicle in sorted(vehicles):
                transform = vehicle.get_transform()
                rotation = transform.rotation

                if d > 200.0:
                    break
                elif d < 60 and info_take == 0:
                    # if is_vehicle_in_front(t, vehicle):
                    info_take = 1


                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        # 切换_show_info的值
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        # 设置通知文本
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        # 如果显示信息
        if self._show_info:
            # 创建一个220xdim[1]的surface，并设置alpha为100
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            # 将info_surface贴到display的(0,0)位置
            display.blit(info_surface, (0, 0))
            # 设置垂直偏移
            v_offset = 4
            # 设置水平偏移
            bar_h_offset = 100
            # 设置进度条宽度
            bar_width = 106
            # 遍历_info_text中的每一项
            for item in self._info_text:
                # 如果垂直偏移加上18大于dim[1]，则跳出循环
                if v_offset + 18 > self.dim[1]:
                    break
                # 如果项是列表
                if isinstance(item, list):
                    # 如果列表长度大于1
                    if len(item) > 1:
                        # 绘制折线图
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    # 将item设置为None
                    item = None
                    # 垂直偏移增加18
                    v_offset += 18
                # 如果项是元组
                elif isinstance(item, tuple):
                    # 如果元组中的第二个元素是布尔值
                    if isinstance(item[1], bool):
                        # 绘制矩形
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        # 绘制进度条
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        # 计算进度条的进度
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        # 如果最小值小于0
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    # 将item设置为元组的第一个元素
                    item = item[0]
                # 如果项是字符串
                if item:
                    # 渲染字符串
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    # 将字符串贴到display的(8,v_offset)位置
                    display.blit(surface, (8, v_offset))
                # 垂直偏移增加18
                v_offset += 18
        # 渲染通知
        self._notifications.render(display)
        # 渲染帮助
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        # 渲染文本
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        # 计算剩余时间
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        # 设置透明度
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        # 渲染到屏幕
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        # 将__doc__中的内容按行分割
        lines = __doc__.split('\n')
        # 设置HelpText的尺寸
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        # 设置HelpText的位置
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        # 创建HelpText的表面
        self.surface = pygame.Surface(self.dim)
        # 填充HelpText的表面
        self.surface.fill((0, 0, 0, 0))
        # 遍历每一行
        for n, line in enumerate(lines):
            # 渲染每一行的文本
            text_texture = self.font.render(line, True, (255, 255, 255))
            # 将文本绘制到HelpText的表面
            self.surface.blit(text_texture, (22, n * 22))
            # 设置不渲染
            self._render = False
        # 设置HelpText的透明度
        self.surface.set_alpha(220)

    def toggle(self):
        # 切换不渲染的状态
        self._render = not self._render

    def render(self, display):
        # 如果需要渲染，则将HelpText绘制到指定的表面
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        # 初始化碰撞传感器
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        # 找到碰撞传感器
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # 我们需要将lambda传递给self的弱引用以避免循环
        # reference.
        weak_self = weakref.ref(self)
        # 监听碰撞事件
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        # 创建一个defaultdict，用于存储碰撞历史
        history = collections.defaultdict(int)
        # 遍历历史记录，将每帧的碰撞强度累加到history中
        for frame, intensity in self.history:
            history[frame] += intensity
        # 返回碰撞历史
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        # 获取当前实例的引用
        self = weak_self()
        # 如果当前实例不存在，则直接返回
        if not self:
            return
        # 获取其他actor的显示名称
        actor_type = get_actor_display_name(event.other_actor)
        # 在hud中显示碰撞信息
        self.hud.notification('Collision with %r' % actor_type)
        # 获取碰撞冲量
        impulse = event.normal_impulse
        # 计算碰撞强度
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        # 将碰撞信息添加到历史记录中
        self.history.append((event.frame, intensity))
        # 如果历史记录超过4000，则删除第一个元素
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================

# 定义压线传感器，检测是否压线
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        # 显示压线通知
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================

# 定义GPS传感器，用于获取车辆的GPS信息，经度和维度
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        # 切换摄像头视角
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        # 设置摄像头的变换矩阵
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# 障碍物
# 用于创建事故场景
def create_accident_scene_bored(client, circle_count):
    # 定义一个空列表，用于存储事故场景
    scene_list = []
    # 计算circle_count除以7的余数
    count = circle_count % 7
    # 如果余数为1
    if count == 1:
        # 调用get_accident_scene_0函数，将结果赋值给scene_list
        scene_list = get_accident_scene_0(client)
    # # 如果余数为2
    # elif count == 2:
    #     # 调用get_accident_scene_1_1函数，将结果赋值给scene_list
    #     scene_list = get_accident_scene_1_1(client)#4_4game     4_2game
    # # 如果余数为3
    # if count == 3:
    #     # 调用get_accident_scene_1_2函数，将结果赋值给scene_list
    #     scene_list = get_accident_scene_1_2(client)#2-1
    # # 如果余数为4
    # elif count == 4:
    #     # 调用get_accident_scene_1_3函数，将结果赋值给scene_list
    #     scene_list = get_accident_scene_1_3(client)
    # # 如果余数为5
    # if count == 5:
    #     # 调用get_accident_scene_1_4函数，将结果赋值给scene_list
    #     scene_list = get_accident_scene_1_4(client)
    # # 如果余数为6
    # elif count == 6:
    #     # 调用get_accident_scene_2_1函数，将结果赋值给scene_list
    #     scene_list = get_accident_scene_2_1(client)
    # # 如果余数为0
    # if count == 0:
    #     # 调用get_accident_scene_3_1函数，将结果赋值给scene_list
    #     scene_list = get_accident_scene_3_1(client)
    # 返回scene_list
    return scene_list


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    # 初始化pygame
    pygame.init()
    pygame.font.init()
    world = None

    try:
        # 连接到carla服务器
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        # 设置pygame显示模式
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        # 初始化hud
        hud = HUD(args.width, args.height)
        # 初始化world
        world = World(client.get_world(), hud, args.filter)
        # 初始化控制器
        # controller = DualControl(world, args.autopilot)
        controller = KeyboardControl(world, args.autopilot)
        # 设置时钟
        clock = pygame.time.Clock()
        #   载入故障点
        circle_count = 1
        scene_list = create_accident_scene_bored(client, circle_count)
        # for i in range(0, 4):
        #     scene_list = create_accident_scene_bored(client, circle_count)
        # print(scene_list)
        last_pos = None
        count = 0
        myfirstime = 0.0
        r = -1
        while True:
            # 设置时钟频率
            clock.tick_busy_loop(60)
            # 解析事件
            # if controller.parse_events(world, clock):
            #     return
            if controller.parse_events(client, world, clock, args.sync):
                return
            # 更新world
            world.tick(clock)
            # 渲染world
            world.render(display)
            # 更新显示
            pygame.display.flip()

            global info_take
            global TOR_cnt
            global max_count
            global success_flag
            if r == -1:
                r = random.randint(0, 11)
                print(r)
            if info_take == 1 or info_take == 2:

                if hud.simulation_time - myfirstime > 0.5:
                    count += 1
                    myfirstime = hud.simulation_time
                    if info_take == 1:
                        # random_int = random.randint(1, 5)
                        # if random_int == 1 or count > 9:
                        if count >= r:
                            print(count)
                            hud.notification('Please takevoer ')
                            # winsound.Beep(1000, 500)  # 发出1000Hz频率，持续500毫秒的提示音
                            # 加载一个声音文件（这里假设你有一个名为'sound.wav'的文件）
                            sound = pygame.mixer.Sound('D:/Carla913/didi.mp3')
                            # 播放声音
                            sound.play()
                            info_take = 2
                    elif info_take == 2:
                        TOR_cnt += 1
                        hud.notification('Please takevoer ')
                        # winsound.Beep(1000, 500)  # 发出1000Hz频率，持续500毫秒的提示音
                        # 加载一个声音文件（这里假设你有一个名为'sound.wav'的文件）
                        sound = pygame.mixer.Sound('D:/Carla913/didi.mp3')
                        # 播放声音
                        sound.play()
                # 假设方向盘的左转和右转映射到轴 0
                # steering_input = float(controller._joystick.get_axis(controller._steer_idx))
                # if (abs(steering_input - 0) > 0.05) and (info_take == 2):
                if info_take == 2:
                    for event in pygame.event.get():
                        if event.key == K_LEFT or event.key == K_RIGHT or event.key == K_UP or event.key == K_DOWN:
                            info_take = 3
                            controller._autopilot_enabled = not controller._autopilot_enabled
                            world.player.set_autopilot(controller._autopilot_enabled)
                            # world.hud.notification('Autopilot %s' % ('On' if controller._autopilot_enabled else 'Off'))
                            world.hud.notification('Takevoer')
                            # winsound.Beep(1000, 500)  # 发出1000Hz频率，持续500毫秒的提示音
            if world.collision_sensor.get_collision_history():
                break

    finally:

        if world is not None:
            world.destroy()

        data = {
            'time': mytime,
            'x': myx,
            'y': myy,
            'v km/h': myv,
            'isTOR': isTOR_list,
            'TOR_cnt': TOR_cnt_list
        }
        df = pd.DataFrame(data)
        # 获取当前时间
        now = datetime.datetime.now()

        # 格式化时间字符串，例如：'2023-07-19_15-30-45'
        # time_str = datetime.date.strftime('%Y-%m-%d_%H-%M-%S')
        time_str = now.strftime('%Y-%m-%d_%H-%M-%S')
        # df.to_excel(f'D:/Carla913/{time_str}.xlsx', index=False)
        # 获取切换后的天气配置
        preset = world._weather_presets[world._weather_index]
        # 显示天气信息
        # world.hud.notification('Weather: %s' % preset[1])
        global n
        # global info_take
        if not world.collision_sensor.get_collision_history() and info_take == 3:
            df.to_excel(f'D:/Carla913/T{n}_S_{preset[1]}_{time_str}.xlsx', index=False)
        else:
            df.to_excel(f'D:/Carla913/T{n}_F_{preset[1]}_{time_str}.xlsx', index=False)
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
