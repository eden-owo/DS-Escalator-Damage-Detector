import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import os
import sys
import time
import argparse
import platform
from ctypes import *

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
Gst.init(None)
import math
from collections import defaultdict
import pyds
import datetime
import threading

MAX_ELEMENTS_IN_DISPLAY_META = 16

SOURCE = ''
CONFIG_INFER_POSE = ''
CONFIG_INFER_DETECT = ''
STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 720
STREAMMUX_HEIGHT = 1280
GPU_ID = 0
PERF_MEASUREMENT_INTERVAL_SEC = 5

try:
    UNTRACKED_OID = pyds.UNTRACKED_OBJECT_ID
except Exception:
    UNTRACKED_OID = 0xFFFFFFFFFFFFFFFF  # DeepStream 未追蹤時常見值

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

start_time = time.time()
fps_streams = {}

CONF_THR = 0.5           # 關鍵點最低信心
DETECT_DEGREE = 45.0     # 觸發門檻（度）
ALARM_KEEP_FRAMES = 30    # 保持紅框的幀數（去抖動）

# 以追蹤 ID 累計/衰減警報的 frame 計數
alarm_counter = defaultdict(int)

# COCO 0-based 索引
IDX = {"LS": 5, "RS": 6, "LH": 11, "RH": 12}

# 事件發送節流（同一個 tracking ID 每 N 幀才送一次，避免刷頻）
EMIT_GAP_FRAMES = 30
_last_emit_frame = {}  # tid -> last_frame_num

# --- fallback: convert signed/py-int to unsigned 64 for DeepStream trackingId ---
def long_to_uint64(val):
    if val is None:
        return 0
    try:
        v = int(val)
    except Exception:
        return 0
    # DeepStream 會用 uint64；未追蹤或負值就設 0
    if v < 0:
        return 0
    return v & 0xFFFFFFFFFFFFFFFF

def _attach_fall_event(batch_meta, frame_meta, obj_meta, body_degree, fall_state):
    """把跌倒事件掛到該 frame 上，讓 nvmsgconv/nvmsgbroker 發出去。"""
    # 1) 先拿一個 user meta 容器
    user_event_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
    if not user_event_meta:
        print("[fall-event] acquire_user_meta_from_pool failed")
        return False

    # 2) 產生 NvDsEventMsgMeta
    msg_meta = pyds.alloc_nvds_event_msg_meta(user_event_meta)

    # 基本欄位
    msg_meta.bbox.top    = obj_meta.rect_params.top
    msg_meta.bbox.left   = obj_meta.rect_params.left
    msg_meta.bbox.width  = obj_meta.rect_params.width
    msg_meta.bbox.height = obj_meta.rect_params.height

    msg_meta.frameId    = frame_meta.frame_num
    msg_meta.trackingId = long_to_uint64(getattr(obj_meta, "object_id", 0))
    msg_meta.confidence = float(max(0.0, min(1.0, obj_meta.confidence)))

    # 補上來源資訊與時間戳
    msg_meta.sensorId  = int(getattr(frame_meta, "source_id", 0))
    msg_meta.placeId   = 0
    msg_meta.moduleId  = 0
    msg_meta.sensorStr = "sensor-{}".format(msg_meta.sensorId)

    msg_meta.ts = pyds.alloc_buffer(33)
    pyds.generate_ts_rfc3339(msg_meta.ts, 32)

    # 設定類型與對象（以 PERSON 為例）
    msg_meta.objType    = pyds.NvDsObjectType.NVDS_OBJECT_TYPE_PERSON
    msg_meta.objClassId = int(getattr(obj_meta, "class_id", 0))
    # 若有 CUSTOM 就用 CUSTOM，否則退回 ENTRY
    try:
        msg_meta.type = pyds.NvDsEventType.NVDS_EVENT_CUSTOM
    except AttributeError:
        msg_meta.type = pyds.NvDsEventType.NVDS_EVENT_ENTRY

    # 3) “零 C++ 變更”的客製：用 NvDsPersonObject 的文字欄位帶出 fall/degree
    person_ext = pyds.alloc_nvds_person_object()
    person_ext = pyds.NvDsPersonObject.cast(person_ext)
    person_ext.gender  = "unknown"
    person_ext.cap     = "fall" if fall_state else "normal"
    person_ext.hair    = "deg={:.2f}".format(body_degree) if body_degree is not None else "deg=NA"
    person_ext.apparel = "fall=1" if fall_state else "fall=0"

    msg_meta.extMsg     = person_ext
    msg_meta.extMsgSize = sys.getsizeof(pyds.NvDsPersonObject)

    # 4) 掛到當前 frame
    user_event_meta.user_meta_data = msg_meta
    user_event_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_EVENT_MSG_META
    pyds.nvds_add_user_meta_to_frame(frame_meta, user_event_meta)
    return True


def _should_emit(frame_num, tracking_id):
    """簡單的 per-tracking 節流，避免每幀都發。未追蹤或 tid<0 就不節流。"""
    if tracking_id <= 0:
        return True
    last = _last_emit_frame.get(tracking_id, -10**9)
    if frame_num - last >= EMIT_GAP_FRAMES:
        _last_emit_frame[tracking_id] = frame_num
        return True
    return False


def angle_between(v1, v2):
    x1,y1 = v1; x2,y2 = v2
    n1 = math.hypot(x1,y1); n2 = math.hypot(x2,y2)
    if n1 == 0 or n2 == 0: return None
    cosv = max(-1.0, min(1.0, (x1*x2 + y1*y2)/(n1*n2)))
    return math.degrees(math.acos(cosv))

def side_tilt_degree(kps, side: str, conf_thr=0.5):
    """side ∈ {'L','R'}，回傳 該側(髖→肩)向量對垂直(0,-1)的角度；若關鍵點缺失則回 None"""
    if side == 'L':
        s, h = IDX["LS"], IDX["LH"]
    else:
        s, h = IDX["RS"], IDX["RH"]
    if s >= len(kps) or h >= len(kps): return None
    xs, ys, cs = kps[s]; xh, yh, ch = kps[h]
    if cs < conf_thr or ch < conf_thr: return None
    v = (xs - xh, ys - yh)              # 髖->肩
    return angle_between(v, (0.0, -1.0))  # 與垂直向上的角度

class GETFPS:
    def __init__(self, stream_id):
        global start_time
        self.start_time = start_time
        self.is_first = True
        self.frame_count = 0
        self.stream_id = stream_id
        self.total_fps_time = 0
        self.total_frame_count = 0

    def get_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        current_time = end_time - self.start_time
        if current_time > PERF_MEASUREMENT_INTERVAL_SEC:
            self.total_fps_time = self.total_fps_time + current_time
            self.total_frame_count = self.total_frame_count + self.frame_count
            current_fps = float(self.frame_count) / current_time
            avg_fps = float(self.total_frame_count) / self.total_fps_time
            sys.stdout.write('DEBUG: FPS of stream %d: %.2f (%.2f)\n' % (self.stream_id + 1, current_fps, avg_fps))
            self.start_time = end_time
            self.frame_count = 0
        else:
            self.frame_count = self.frame_count + 1


def set_custom_bbox(obj_meta):
    border_width = 0 #6 
    font_size = 0 #18
    x_offset = int(min(STREAMMUX_WIDTH - 1, max(0, obj_meta.rect_params.left - (border_width / 2))))
    y_offset = int(min(STREAMMUX_HEIGHT - 1, max(0, obj_meta.rect_params.top - (font_size * 2) + 1)))

    obj_meta.rect_params.border_width = border_width
    obj_meta.rect_params.border_color.red = 0.0
    obj_meta.rect_params.border_color.green = 0.0
    obj_meta.rect_params.border_color.blue = 1.0
    obj_meta.rect_params.border_color.alpha = 1.0
    obj_meta.text_params.font_params.font_name = 'Ubuntu'
    obj_meta.text_params.font_params.font_size = font_size
    obj_meta.text_params.x_offset = x_offset
    obj_meta.text_params.y_offset = y_offset
    obj_meta.text_params.font_params.font_color.red = 1.0
    obj_meta.text_params.font_params.font_color.green = 1.0
    obj_meta.text_params.font_params.font_color.blue = 1.0
    obj_meta.text_params.font_params.font_color.alpha = 1.0
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.red = 0.0
    obj_meta.text_params.text_bg_clr.green = 0.0
    obj_meta.text_params.text_bg_clr.blue = 1.0
    obj_meta.text_params.text_bg_clr.alpha = 1.0


def parse_pose_from_meta(frame_meta, obj_meta):
    # 讀取關鍵點數量
    num_joints = int(obj_meta.mask_params.size / (sizeof(c_float) * 3))

    # 還原 letterbox（把模型輸入座標轉回原圖座標）
    gain = min(obj_meta.mask_params.width / STREAMMUX_WIDTH,
               obj_meta.mask_params.height / STREAMMUX_HEIGHT)
    pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) / 2.0
    pad_y = (obj_meta.mask_params.height - STREAMMUX_HEIGHT * gain) / 2.0

    # 向 DeepStream 的 display meta 池子要一個 display_meta 來畫圖形
    batch_meta = frame_meta.base_meta.batch_meta
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    # 第一個 for：畫「關鍵點（圓）」
    for i in range(num_joints):
        data = obj_meta.mask_params.get_mask_array()
        xc = int((data[i * 3 + 0] - pad_x) / gain)
        yc = int((data[i * 3 + 1] - pad_y) / gain)
        confidence = data[i * 3 + 2]

        if confidence < 0.5:
            continue

        # 滿量就換新 display_meta：
        if display_meta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # 設定這個點的繪圖參數並遞增計數
        circle_params = display_meta.circle_params[display_meta.num_circles]
        circle_params.xc = xc
        circle_params.yc = yc
        circle_params.radius = 6
        circle_params.circle_color.red = 1.0
        circle_params.circle_color.green = 1.0
        circle_params.circle_color.blue = 1.0
        circle_params.circle_color.alpha = 1.0
        circle_params.has_bg_color = 1
        circle_params.bg_color.red = 0.0
        circle_params.bg_color.green = 0.0
        circle_params.bg_color.blue = 1.0
        circle_params.bg_color.alpha = 1.0
        display_meta.num_circles += 1
    
    # 第二個 for：畫「骨架連線（線）」
    for i in range(num_joints + 2):
        data = obj_meta.mask_params.get_mask_array()
        x1 = int((data[(skeleton[i][0] - 1) * 3 + 0] - pad_x) / gain)
        y1 = int((data[(skeleton[i][0] - 1) * 3 + 1] - pad_y) / gain)
        confidence1 = data[(skeleton[i][0] - 1) * 3 + 2]
        x2 = int((data[(skeleton[i][1] - 1) * 3 + 0] - pad_x) / gain)
        y2 = int((data[(skeleton[i][1] - 1) * 3 + 1] - pad_y) / gain)
        confidence2 = data[(skeleton[i][1] - 1) * 3 + 2]

        if confidence1 < 0.5 or confidence2 < 0.5:
            continue

        if display_meta.num_lines == MAX_ELEMENTS_IN_DISPLAY_META:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        line_params = display_meta.line_params[display_meta.num_lines]
        line_params.x1 = x1
        line_params.y1 = y1
        line_params.x2 = x2
        line_params.y2 = y2
        line_params.line_width = 6
        line_params.line_color.red = 0.0
        line_params.line_color.green = 0.0
        line_params.line_color.blue = 1.0
        line_params.line_color.alpha = 1.0
        display_meta.num_lines += 1



def extract_keypoints(obj_meta, stream_w, stream_h, conf_thr=None):
    """
    從 obj_meta.mask_params 取回 (x, y, conf) 關鍵點，並轉到 nvstreammux 的座標系。
    - stream_w/stream_h: 你的 STREAMMUX_WIDTH/HEIGHT
    - conf_thr: 若給值（例如 0.5），會先把低信心點濾掉
    回傳: list[(x: float, y: float, conf: float)]
    """
    mp = obj_meta.mask_params
    if mp.size == 0:
        return []

    data = mp.get_mask_array()  # 連續 float: x0,y0,c0, x1,y1,c1, ...
    num_floats = mp.size // sizeof(c_float)
    if num_floats % 3 != 0:
        # 不是 (x,y,conf)*N 這種結構就放棄
        return []
    num_joints = num_floats // 3

    # 反 letterbox：把模型座標還原到 streammux 影像座標
    if stream_w == 0 or stream_h == 0:
        return []
    gain = min(mp.width / float(stream_w), mp.height / float(stream_h))
    if gain <= 0:
        return []
    pad_x = (mp.width  - stream_w * gain) / 2.0
    pad_y = (mp.height - stream_h * gain) / 2.0

    kps = []
    for i in range(num_joints):
        rx = float(data[i*3 + 0])  # 模型座標 x
        ry = float(data[i*3 + 1])  # 模型座標 y
        c  = float(data[i*3 + 2])

        if conf_thr is not None and c < conf_thr:
            continue

        x = (rx - pad_x) / gain
        y = (ry - pad_y) / gain

        # 夾在影像範圍（避免 OSD 越界）
        x = max(0.0, min(float(stream_w  - 1), x))
        y = max(0.0, min(float(stream_h - 1), y))

        # 保留浮點精度；真正畫圖時再 int()
        kps.append((x, y, c))

    return kps



def draw_pose_and_get_kps(frame_meta, obj_meta):
    # 直接沿用你原本的畫點 / 畫線邏輯，只是先把 kps 算好回傳
    kps = extract_keypoints(obj_meta, STREAMMUX_WIDTH, STREAMMUX_HEIGHT, conf_thr=0)

    batch_meta = frame_meta.base_meta.batch_meta
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    # 畫關鍵點
    # for (xc, yc, c) in kps:
    #     if c < 0.5:
    #         continue
    #     if display_meta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
    #         display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    #         pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    #     circle_params = display_meta.circle_params[display_meta.num_circles]
    #     circle_params.xc = int(xc)
    #     circle_params.yc = int(yc)
    #     circle_params.radius = 6
    #     circle_params.circle_color.red = 1.0
    #     circle_params.circle_color.green = 1.0
    #     circle_params.circle_color.blue = 1.0
    #     circle_params.circle_color.alpha = 1.0
    #     circle_params.has_bg_color = 1
    #     circle_params.bg_color.red = 0.0
    #     circle_params.bg_color.green = 0.0
    #     circle_params.bg_color.blue = 1.0
    #     circle_params.bg_color.alpha = 1.0
    #     display_meta.num_circles += 1

    # # 畫骨架線段（沿用你的 skeleton）
    # for a, b in skeleton:
    #     ia, ib = a - 1, b - 1
    #     if ia >= len(kps) or ib >= len(kps):
    #         continue
    #     x1, y1, c1 = kps[ia]
    #     x2, y2, c2 = kps[ib]
    #     if c1 < 0.5 or c2 < 0.5:
    #         continue
    #     if display_meta.num_lines == MAX_ELEMENTS_IN_DISPLAY_META:
    #         display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    #         pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    #     line_params = display_meta.line_params[display_meta.num_lines]
    #     line_params.x1 = int(x1); line_params.y1 = int(y1)
    #     line_params.x2 = int(x2); line_params.y2 = int(y2)
    #     line_params.line_width = 6
    #     line_params.line_color.red = 0.0
    #     line_params.line_color.green = 0.0
    #     line_params.line_color.blue = 1.0
    #     line_params.line_color.alpha = 1.0
    #     display_meta.num_lines += 1

    return kps


# ====== [ADD] 命中條件時覆蓋紅色骨架（節省元素：預設只畫線，不重畫點） ======
def overlay_pose_with_color(frame_meta, kps, line_rgb=(1.0, 0.0, 0.0), draw_points=False, point_rgb=(1.0, 1.0, 1.0)):
    batch_meta = frame_meta.base_meta.batch_meta
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    if draw_points:
        for (x, y, c) in kps:
            if c < CONF_THR:
                continue
            if display_meta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            cp = display_meta.circle_params[display_meta.num_circles]
            cp.xc = int(x); cp.yc = int(y); cp.radius = 6
            cp.circle_color.red, cp.circle_color.green, cp.circle_color.blue = point_rgb
            cp.circle_color.alpha = 1.0
            cp.has_bg_color = 0
            display_meta.num_circles += 1

    for a, b in skeleton:
        ia, ib = a-1, b-1
        if ia >= len(kps) or ib >= len(kps): 
            continue
        x1, y1, c1 = kps[ia]; x2, y2, c2 = kps[ib]
        if c1 < CONF_THR or c2 < CONF_THR:
            continue
        if display_meta.num_lines == MAX_ELEMENTS_IN_DISPLAY_META:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        lp = display_meta.line_params[display_meta.num_lines]
        lp.x1 = int(x1); lp.y1 = int(y1)
        lp.x2 = int(x2); lp.y2 = int(y2)
        lp.line_width = 6
        lp.line_color.red, lp.line_color.green, lp.line_color.blue = line_rgb
        lp.line_color.alpha = 1.0
        display_meta.num_lines += 1
# ====== [ADD] 結束 ======


def tracker_src_pad_buffer_probe(pad, info, user_data):
    buf = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        current_index = frame_meta.source_id
        fall_state = 0   # 預設沒跌倒

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            kps = draw_pose_and_get_kps(frame_meta, obj_meta)
            set_custom_bbox(obj_meta)
            
            # ====== [ADD] 套用 body_degree 判斷 + 紅色覆蓋 ======
            left_deg  = side_tilt_degree(kps, 'L', CONF_THR)
            right_deg = side_tilt_degree(kps, 'R', CONF_THR)
            left_line  = 1 if left_deg  is not None else 0
            right_line = 1 if right_deg is not None else 0

            if left_line == 0 and right_line == 0:
                body_degree = None
            elif left_line == 0:
                body_degree = abs(right_deg)
            elif right_line == 0:
                body_degree = abs(left_deg)
            else:
                body_degree = abs(left_deg + right_deg) / 2.0

            # 追蹤 ID（未追蹤則 -1：只做即時，不做累積）
            oid = int(getattr(obj_meta, "object_id", -1))
            tid = oid if oid not in (UNTRACKED_OID, 0, -1) else -1

            # 累積/衰減警報保持幀
            if body_degree is not None and body_degree >= DETECT_DEGREE:
                if tid != -1:
                    alarm_counter[tid] = ALARM_KEEP_FRAMES
            else:
                if tid != -1:
                    alarm_counter[tid] = max(0, alarm_counter[tid] - 1)

            triggered = (body_degree is not None and body_degree >= DETECT_DEGREE) or (tid != -1 and alarm_counter[tid] > 0)
            if triggered:
                fall_state = 1
                # 覆蓋紅框
                obj_meta.rect_params.border_color.red   = 1.0
                obj_meta.rect_params.border_color.green = 0.0
                obj_meta.rect_params.border_color.blue  = 0.0
                obj_meta.rect_params.border_color.alpha = 1.0
                obj_meta.rect_params.border_width = 8

                # 疊一層紅色骨架（線），點可省略以節省元素
                overlay_pose_with_color(frame_meta, kps, line_rgb=(1.0, 0.0, 0.0), draw_points=False)

                # 顯示角度文字
                if body_degree is not None:
                    obj_meta.text_params.display_text = f'degree: {body_degree:.2f}'
                    obj_meta.text_params.font_params.font_name = 'Ubuntu'
                    obj_meta.text_params.font_params.font_size = 18
                    obj_meta.text_params.font_params.font_color.red   = 1.0
                    obj_meta.text_params.font_params.font_color.green = 1.0
                    obj_meta.text_params.font_params.font_color.blue  = 1.0
                    obj_meta.text_params.font_params.font_color.alpha = 1.0
                    obj_meta.text_params.set_bg_clr = 1
                    obj_meta.text_params.text_bg_clr.red   = 1.0
                    obj_meta.text_params.text_bg_clr.green = 0.0
                    obj_meta.text_params.text_bg_clr.blue  = 0.0
                    obj_meta.text_params.text_bg_clr.alpha = 1.0
            
                # ===================== 這裡是新增的「事件掛載」 =====================
                # 有追蹤 ID 就用它節流，沒有就不節流
                tid_for_emit = tid if tid != -1 else 0
                if _should_emit(frame_meta.frame_num, tid_for_emit):
                    ok = _attach_fall_event(
                        batch_meta=batch_meta,
                        frame_meta=frame_meta,
                        obj_meta=obj_meta,
                        body_degree=body_degree,
                        fall_state=True
                    )
                    if not ok:
                        print("[fall-event] attach failed")
                # ====================================================================
            # ====== [ADD] 結束 ======

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # === 只要有跌倒，發送事件 ===
        # if fall_state == 1:
        #     print(" ")
        fps_streams['stream{0}'.format(current_index)].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find('decodebin') != -1:
        Object.connect('child-added', decodebin_child_added, user_data)
    if name.find('nvv4l2decoder') != -1:
        Object.set_property('drop-frame-interval', 0)
        Object.set_property('num-extra-surfaces', 1)
        if is_aarch64():
            Object.set_property('enable-max-performance', 1)
        else:
            Object.set_property('cudadec-memtype', 0)
            Object.set_property('gpu-id', GPU_ID)


def cb_newpad(decodebin, pad, user_data):
    streammux_sink_pad = user_data
    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps()
    structure = caps.get_structure(0)
    name = structure.get_name()
    features = caps.get_features(0)
    if name.find('video') != -1:
        if features.contains('memory:NVMM'):
            if pad.link(streammux_sink_pad) != Gst.PadLinkReturn.OK:
                sys.stderr.write('ERROR: Failed to link source to streammux sink pad\n')
        else:
            sys.stderr.write('ERROR: decodebin did not pick NVIDIA decoder plugin')


def create_uridecode_bin(stream_id, uri, streammux):
    bin_name = 'source-bin-%04d' % stream_id
    bin = Gst.ElementFactory.make('uridecodebin', bin_name)
    if 'rtsp://' in uri:
        pyds.configure_source_for_ntp_sync(hash(bin))
    bin.set_property('uri', uri)
    pad_name = 'sink_%u' % stream_id
    streammux_sink_pad = streammux.get_request_pad(pad_name)
    bin.connect('pad-added', cb_newpad, streammux_sink_pad)
    bin.connect('child-added', decodebin_child_added, 0)
    fps_streams['stream{0}'.format(stream_id)] = GETFPS(stream_id)
    return bin


def bus_call(bus, message, user_data):
    loop = user_data
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write('DEBUG: EOS\n')
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write('WARNING: %s: %s\n' % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write('ERROR: %s: %s\n' % (err, debug))
        loop.quit()
    return True


def is_aarch64():
    return platform.uname()[4] == 'aarch64'




def main():
    loop = GLib.MainLoop()
    pipeline = Gst.Pipeline()

    streammux = Gst.ElementFactory.make('nvstreammux', 'nvstreammux')
    if not streammux:
        sys.stderr.write('ERROR: Failed to create nvstreammux\n')
        sys.exit(1)

    source_bin = create_uridecode_bin(0, SOURCE, streammux)
    if not source_bin:
        sys.stderr.write('ERROR: Failed to create source_bin\n')
        sys.exit(1)

    pgie_pose = Gst.ElementFactory.make('nvinfer', 'pgie_pose')
    pgie_det  = Gst.ElementFactory.make('nvinfer', 'pgie_det')
    if not pgie_pose or not pgie_det:
        sys.stderr.write('ERROR: Failed to create nvinfer\n')
        sys.exit(1)

    tracker = Gst.ElementFactory.make('nvtracker', 'nvtracker')
    if not tracker:
        sys.stderr.write('ERROR: Failed to create nvtracker\n')
        sys.exit(1)

    tee_main = Gst.ElementFactory.make("tee", "tee_main")
    if not tee_main:
        sys.stderr.write("ERROR: Failed to create tee_main\n"); sys.exit(1)

    queue_pose = Gst.ElementFactory.make("queue", "queue_pose_infer")
    queue_det  = Gst.ElementFactory.make("queue", "queue_det_infer")
    if not queue_pose or not queue_det:
        sys.stderr.write("ERROR: Failed to create infer queues\n"); sys.exit(1)

    tee = Gst.ElementFactory.make("tee", "tee")
    if not tee:
        sys.stderr.write("ERROR: Failed to create tee\n")
        sys.exit(1)

    queue_osd = Gst.ElementFactory.make("queue", "queue_osd")
    if not queue_osd:
        sys.stderr.write("ERROR: Failed to create queue_osd\n")
        sys.exit(1)

    converter = Gst.ElementFactory.make('nvvideoconvert', 'nvvideoconvert')
    if not converter:
        sys.stderr.write('ERROR: Failed to create nvvideoconvert\n')
        sys.exit(1)

    osd = Gst.ElementFactory.make('nvdsosd', 'nvdsosd')
    if not osd:
        sys.stderr.write('ERROR: Failed to create nvdsosd\n')
        sys.exit(1)

    sink = None
    if is_aarch64():
        sink = Gst.ElementFactory.make('nv3dsink', 'nv3dsink')
        if not sink:
            sys.stderr.write('ERROR: Failed to create nv3dsink\n')
            sys.exit(1)
    else:
        sink = Gst.ElementFactory.make('nveglglessink', 'nveglglessink')
        if not sink:
            sys.stderr.write('ERROR: Failed to create nveglglessink\n')
            sys.exit(1)

    queue_det_out  = Gst.ElementFactory.make("queue", "queue_det_out")
    fakesink_det   = Gst.ElementFactory.make("fakesink", "fakesink_det")
    if not queue_det_out or not fakesink_det:
        sys.stderr.write("ERROR: Failed to create det tail (queue/fakesink)\n"); sys.exit(1)

    # 建議關掉同步/最後樣本，避免多餘負擔
    fakesink_det.set_property("sync", False)
    fakesink_det.set_property("async", False)
    fakesink_det.set_property("qos", 0)
    fakesink_det.set_property("enable-last-sample", False)

    # === msg pipeline ===
    queue_msg = Gst.ElementFactory.make("queue", "queue_msg")
    queue_msg.set_property("leaky", 2)
    queue_msg.set_property("max-size-buffers", 1)
    if not queue_msg:
        sys.stderr.write("ERROR: Failed to create queue_msg\n")
        sys.exit(1)

    # 假設之後要接 msgconv / msgbroker，就可以這裡建立
    msgconv = Gst.ElementFactory.make("nvmsgconv", "msgconv")
    if not msgconv:
        sys.stderr.write("ERROR: Failed to create msgconv\n")
        sys.exit(1)

    msgbroker = Gst.ElementFactory.make("nvmsgbroker", "msgbroker")
    if not msgbroker:
        sys.stderr.write("ERROR: Failed to create msgbroker\n")
        sys.exit(1)


    sys.stdout.write('\n')
    sys.stdout.write('SOURCE: %s\n' % SOURCE)
    sys.stdout.write('CONFIG_INFER_POSE: %s\n' % CONFIG_INFER_POSE)
    sys.stdout.write('CONFIG_INFER_DETECT: %s\n' % CONFIG_INFER_DETECT)
    sys.stdout.write('STREAMMUX_BATCH_SIZE: %d\n' % STREAMMUX_BATCH_SIZE)
    sys.stdout.write('STREAMMUX_WIDTH: %d\n' % STREAMMUX_WIDTH)
    sys.stdout.write('STREAMMUX_HEIGHT: %d\n' % STREAMMUX_HEIGHT)
    sys.stdout.write('GPU_ID: %d\n' % GPU_ID)
    sys.stdout.write('PERF_MEASUREMENT_INTERVAL_SEC: %d\n' % PERF_MEASUREMENT_INTERVAL_SEC)
    sys.stdout.write('JETSON: %s\n' % ('TRUE' if is_aarch64() else 'FALSE'))
    sys.stdout.write('\n')

    streammux.set_property('batch-size', STREAMMUX_BATCH_SIZE)
    streammux.set_property('batched-push-timeout', 25000)
    streammux.set_property('width', STREAMMUX_WIDTH)
    streammux.set_property('height', STREAMMUX_HEIGHT)
    streammux.set_property('enable-padding', 1)
    streammux.set_property('live-source', 1)
    streammux.set_property('attach-sys-ts', 1)
    pgie_pose.set_property('config-file-path', CONFIG_INFER_POSE)
    pgie_pose.set_property('qos', 0)
    pgie_det.set_property('config-file-path', CONFIG_INFER_DETECT)
    pgie_det.set_property('qos', 0)    
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file',
                         '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')
    tracker.set_property('display-tracking-id', 1)
    tracker.set_property('qos', 0)
    osd.set_property('process-mode', int(pyds.MODE_GPU))
    osd.set_property('qos', 0)
    sink.set_property('async', 0)
    sink.set_property('sync', 0)
    sink.set_property('qos', 0)

    if 'file://' in SOURCE:
        streammux.set_property('live-source', 0)

    if tracker.find_property('enable_batch_process') is not None:
        tracker.set_property('enable_batch_process', 1)

    if tracker.find_property('enable_past_frame') is not None:
        tracker.set_property('enable_past_frame', 1)

    msgbroker.set_property("proto-lib", "/opt/nvidia/deepstream/deepstream/lib/libnvds_mqtt_proto.so")
    msgbroker.set_property("sync", False)

    # msgbroker.set_property("conn-str", "localhost;1883;ds/events")  # broker_ip;port;topic

    msgbroker.set_property("conn-str", "mosq;1883;psmt-fall-pub")  # host;port;clientid
    msgbroker.set_property("topic", "ds/events")                  # topic 分開指定


    if not is_aarch64():
        streammux.set_property('nvbuf-memory-type', 0)
        streammux.set_property('gpu_id', GPU_ID)
        pgie_pose.set_property('gpu_id', GPU_ID)
        pgie_det.set_property('gpu_id', GPU_ID)
        tracker.set_property('gpu_id', GPU_ID)
        converter.set_property('nvbuf-memory-type', 0)
        converter.set_property('gpu_id', GPU_ID)
        osd.set_property('gpu_id', GPU_ID)

    # 加入 pipeline
    for elem in [
                streammux, source_bin,
                tee_main, queue_pose, pgie_pose,
                queue_det,  pgie_det,
                tracker,  # 單一 tracker（接在 pose 分支後）
                tee, queue_osd, converter, osd, sink,
                queue_msg, msgconv, msgbroker, 
                queue_det_out, fakesink_det]:
        pipeline.add(elem)

    if not streammux.link(tee_main):
        sys.stderr.write('ERROR: link streammux→tee_main failed\n'); sys.exit(1)

    # tee_main 出兩個 request pad，分別接到兩個 queue
    tee_pad_pose = tee_main.get_request_pad("src_%u")
    tee_pad_det  = tee_main.get_request_pad("src_%u")
    q_pose_sink  = queue_pose.get_static_pad("sink")
    q_det_sink   = queue_det.get_static_pad("sink")
    if not (tee_pad_pose and tee_pad_det and q_pose_sink and q_det_sink):
        sys.stderr.write('ERROR: tee_main or infer queue pads missing\n'); sys.exit(1)
    tee_pad_pose.link(q_pose_sink)
    tee_pad_det.link(q_det_sink)

    # === Pose 分支（主幹）：queue_pose → pgie_pose → tracker → tee（你的 OSD/msg 分支沿用）
    if not queue_pose.link(pgie_pose): sys.exit(1)
    if not pgie_pose.link(tracker):    sys.exit(1)
    if not tracker.link(tee):          sys.exit(1)

    # === osd pipeline ===
    # tee → queue_osd → converter → osd → sink
    tee_osd_pad = tee.get_request_pad("src_%u")
    queue_osd_sink_pad = queue_osd.get_static_pad("sink")
    if not tee_osd_pad or not queue_osd_sink_pad:
        sys.stderr.write("ERROR: Unable to get tee or queue pads\n")
        sys.exit(1)

    tee_osd_pad.link(queue_osd_sink_pad)

    queue_osd.link(converter)
    converter.link(osd)
    osd.link(sink)
    # === osd pipeline ===

    # === msg pipeline ===
    # tee → queue_msg → msgconv → msgbroker
    tee_msg_pad = tee.get_request_pad("src_%u")
    # queue_msg.set_property("leaky", 2)        # 丟掉舊 buffer，防止阻塞
    # queue_msg.set_property("max-size-buffers", 1)
    queue_msg_sink_pad = queue_msg.get_static_pad("sink")
    if not tee_msg_pad or not queue_msg_sink_pad:
        sys.stderr.write("ERROR: Unable to get tee/msg queue pads\n")
        sys.exit(1)
    
    msgconv.set_property("config", "/apps/msgconv/msgconv_config.txt")
    msgconv.set_property("payload-type", 1)  # 0 = DeepStream schema, 1 = minimal schema

    tee_msg_pad.link(queue_msg_sink_pad)

    queue_msg.link(msgconv)
    msgconv.link(msgbroker)
    # === msg pipeline ===    

    # 監聽 pipeline 的 bus，處理錯誤/結束訊息    
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', bus_call, loop)

    # === Det 分支（不接 tracker）：queue_det → pgie_det（src 掛收集 probe）===
    if not queue_det.link(pgie_det): sys.exit(1)
    if not pgie_det.link(queue_det_out): sys.exit(1)
    if not queue_det_out.link(fakesink_det): sys.exit(1)
    # ---- 掛 probe：收集 det 物件、在 tracker 前合併 ----
    _det_cache = {}
    _det_lock  = threading.Lock()

    def collect_det_probe(pad, info, udata):
        buf = info.get_buffer()
        if not buf: return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            fmeta = pyds.NvDsFrameMeta.cast(l_frame.data)
            dets = []
            l_obj = fmeta.obj_meta_list
            while l_obj:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                dets.append((
                    obj.rect_params.left, obj.rect_params.top,
                    obj.rect_params.width, obj.rect_params.height,
                    obj.class_id, obj.confidence
                ))
                l_obj = l_obj.next
            if dets:
                with _det_lock:
                    _det_cache[(fmeta.source_id, fmeta.frame_num)] = dets
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK

    def merge_before_tracker_probe(pad, info, udata):
        buf = info.get_buffer()
        if not buf: return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            fmeta = pyds.NvDsFrameMeta.cast(l_frame.data)
            key = (fmeta.source_id, fmeta.frame_num)
            with _det_lock:
                dets = _det_cache.pop(key, None)
            if dets:
                for (x, y, w, h, cls, conf) in dets:
                    om = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                    om.rect_params.left = x; om.rect_params.top = y
                    om.rect_params.width = w; om.rect_params.height = h
                    om.class_id = cls; om.confidence = conf
                    om.unique_component_id = 2  # 來自 pgie_det
                    pyds.nvds_add_obj_meta_to_frame(fmeta, om, None)
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK

    pgie_det_src = pgie_det.get_static_pad('src')
    pgie_det_src.add_probe(Gst.PadProbeType.BUFFER, collect_det_probe, 0)

    pgie_pose_src = pgie_pose.get_static_pad('src')  # tracker 前
    pgie_pose_src.add_probe(Gst.PadProbeType.BUFFER, merge_before_tracker_probe, 0)

    # 在 tracker 的 src pad 加 probe，方便存取 metadata (例如偵測框/追蹤資訊)
    tracker_src_pad = tracker.get_static_pad('src')
    if not tracker_src_pad:
        sys.stderr.write('ERROR: Failed to get tracker src pad\n')
        sys.exit(1)
    else:
        tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0)

    # 啟動 pipeline，進入主迴圈 (loop.run)，直到 EOS 或錯誤才結束
    pipeline.set_state(Gst.State.PLAYING)
    sys.stdout.write('\n')
    try:
        loop.run()
    except:
        pass
    pipeline.set_state(Gst.State.NULL)
    sys.stdout.write('\n')


def parse_args():
    global SOURCE, CONFIG_INFER_POSE, CONFIG_INFER_DETECT, STREAMMUX_BATCH_SIZE, STREAMMUX_WIDTH, STREAMMUX_HEIGHT, GPU_ID, \
        PERF_MEASUREMENT_INTERVAL_SEC

    parser = argparse.ArgumentParser(description='DeepStream')
    parser.add_argument('-s', '--source', required=True, help='Source stream/file')
    parser.add_argument('-cip', '--config-infer-pose', required=True, help='Config infer file (POSE)')
    parser.add_argument('-cid', '--config-infer-detect', required=True, help='Config infer file (Detect)')
    parser.add_argument('-b', '--streammux-batch-size', type=int, default=1, help='Streammux batch-size (default: 1)')
    parser.add_argument('-w', '--streammux-width', type=int, default=720, help='Streammux width (default: 1920)')
    parser.add_argument('-e', '--streammux-height', type=int, default=1280, help='Streammux height (default: 1080)')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, help='GPU id (default: 0)')
    parser.add_argument('-f', '--fps-interval', type=int, default=5, help='FPS measurement interval (default: 5)')
    args = parser.parse_args()
    if args.source == '':
        sys.stderr.write('ERROR: Source not found\n')
        sys.exit(1)
    if args.config_infer_pose == '' or not os.path.isfile(args.config_infer_pose):
        sys.stderr.write('ERROR: Config infer (POSE) not found\n')
        sys.exit(1)
    if args.config_infer_detect == '' or not os.path.isfile(args.config_infer_detect):
        sys.stderr.write('ERROR: Config infer (DETECT) not found\n')
        sys.exit(1)        

    SOURCE = args.source
    CONFIG_INFER_DETECT = args.config_infer_detect
    CONFIG_INFER_POSE = args.config_infer_pose
    STREAMMUX_BATCH_SIZE = args.streammux_batch_size
    STREAMMUX_WIDTH = args.streammux_width
    STREAMMUX_HEIGHT = args.streammux_height
    GPU_ID = args.gpu_id
    PERF_MEASUREMENT_INTERVAL_SEC = args.fps_interval


if __name__ == '__main__':
    parse_args()
    sys.exit(main())