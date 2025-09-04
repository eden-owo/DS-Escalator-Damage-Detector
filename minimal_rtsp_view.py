#!/usr/bin/env python3
import gi
import time
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import os, sys, argparse, platform
Gst.init(None)
import pyds
from ctypes import *
import signal

STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080
MAX_ELEMENTS_IN_DISPLAY_META = 16

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

start_time = time.time()
fps_streams = {}

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
    return platform.uname()[4] == "aarch64"

def make(name, alias=None, required=True):
    e = Gst.ElementFactory.make(name, alias or name)
    if required and not e:
        raise RuntimeError(f"Failed to create {name}")
    return e

def decodebin_child_added(_, obj, name, __):
    if "nvv4l2decoder" in name:
        try:
            if is_aarch64():
                obj.set_property("enable-max-performance", True)
            else:
                obj.set_property("gpu-id", 0)
                obj.set_property("cudadec-memtype", 0)
        except Exception:
            pass

def build_source_bin(stream_id: int, uri: str) -> Gst.Element:
    """
    source-bin-%04d:
      uridecodebin (動態 pad)
        → queue_pre
        → nvvideoconvert/nvvidconv/videoconvert
        → capsfilter "video/x-raw(memory:NVMM), format=NV12"
        → queue_post
        → (GhostPad 'src' 暴露出去，靜態)
    """
    bin_name = f"source-bin-{stream_id:04d}"
    bin = make("bin", bin_name)

    uridec = make("uridecodebin", "uridecodebin")
    # 統一補成 URI
    if uri.startswith("/"):
        uri = f"file://{os.path.abspath(uri)}"
    uridec.set_property("uri", uri)
    if uri.startswith("rtsp://"):
        # 降延遲、TCP 比較穩
        try:
            uridec.set_property("buffer-size", 0)
            uridec.set_property("latency", 100)
            uridec.set_property("drop-on-latency", True)
        except Exception:
            pass

    queue_pre  = make("queue",  "queue_pre")
    conv = (Gst.ElementFactory.make("nvvideoconvert", "preconv")
            or Gst.ElementFactory.make("nvvidconv", "preconv")
            or make("videoconvert", "preconv"))
    capsfilter = make("capsfilter", "precap")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    capsfilter.set_property("caps", caps)
    queue_post = make("queue", "queue_post")

    # 加進 bin
    for e in (uridec, queue_pre, conv, capsfilter, queue_post):
        bin.add(e)

    # 動態 pad：只負責把 uridecodebin 的 video pad → queue_pre.sink
    def on_pad_added(src, new_pad):
        caps = new_pad.get_current_caps() or new_pad.query_caps()
        if not caps or caps.get_size() == 0:
            return
        s = caps.get_structure(0)
        if not s:
            return
        name = s.get_name()
        media = s.get_string("media") if s.has_field("media") else None
        if name != "application/x-rtp" and "video" not in name and media != "video":
            return
        sinkpad = queue_pre.get_static_pad("sink")
        if sinkpad and not sinkpad.is_linked():
            new_pad.link(sinkpad)
    uridec.connect("pad-added", on_pad_added)
    uridec.connect("child-added", decodebin_child_added, None)

    # 靜態連結：queue_pre → conv → caps → queue_post
    assert queue_pre.link(conv)
    assert conv.link(capsfilter)
    assert capsfilter.link(queue_post)

    # 建一個 Ghost Pad 當作 bin 的 "src"
    srcpad = queue_post.get_static_pad("src")
    ghost_src = Gst.GhostPad.new("src", srcpad)
    ghost_src.set_active(True)
    bin.add_pad(ghost_src)

    return bin

def parse_pose_from_meta(frame_meta, obj_meta):
    num_joints = int(obj_meta.mask_params.size / (sizeof(c_float) * 3))

    gain = min(obj_meta.mask_params.width / STREAMMUX_WIDTH,
               obj_meta.mask_params.height / STREAMMUX_HEIGHT)
    pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) / 2.0
    pad_y = (obj_meta.mask_params.height - STREAMMUX_HEIGHT * gain) / 2.0

    batch_meta = frame_meta.base_meta.batch_meta
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    for i in range(num_joints):
        data = obj_meta.mask_params.get_mask_array()
        xc = int((data[i * 3 + 0] - pad_x) / gain)
        yc = int((data[i * 3 + 1] - pad_y) / gain)
        confidence = data[i * 3 + 2]

        if confidence < 0.5:
            continue

        if display_meta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

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

def set_custom_bbox(obj_meta):
    border_width = 6
    font_size = 18
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



# 暫時放檔案頂部（import 後）
# def tracker_src_pad_buffer_probe(pad, info, user_data):
#     return Gst.PadProbeReturn.OK

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

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            parse_pose_from_meta(frame_meta, obj_meta)
            set_custom_bbox(obj_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # fps_streams['stream{0}'.format(current_index)].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def build_pipeline_deepstream(uri: str, width=1920, height=1080, batch_size=1, sync=False,
                              config_infer=None, gpu_id=0):
    pipeline = make("pipeline", "deepstream-pipeline")

    # mux
    streammux = make("nvstreammux", "nvstreammux")
    streammux.set_property("batch-size", batch_size)
    streammux.set_property("width", width)
    streammux.set_property("height", height)
    streammux.set_property("batched-push-timeout", 25000)  # us
    streammux.set_property("live-source", int(uri.startswith("rtsp://")))
    streammux.set_property("attach-sys-ts", True)
    streammux.set_property("enable-padding", 1)

    # === DeepStream 模組：nvinfer（pgie）/ nvtracker / nvdsosd ===
    if not config_infer or not os.path.isfile(config_infer):
        raise RuntimeError("請提供有效的 --config-infer 檔案路徑")
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    pgie.set_property("config-file-path", config_infer)

    tracker = Gst.ElementFactory.make("nvtracker", "nvtracker")
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file",
                         "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")

    # 後處理轉 RGBA（給 nvdsosd 用）
    postconv = (Gst.ElementFactory.make("nvvideoconvert", "postconv")
                or Gst.ElementFactory.make("nvvidconv", "postconv")
                or make("videoconvert", "postconv"))
    to_rgba = make("capsfilter", "to_rgba")
    to_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    osd = Gst.ElementFactory.make("nvdsosd", "nvdsosd")
    # 不依賴 pyds 常數，直接設 GPU 模式（等價於 pyds.MODE_GPU）
    if osd.find_property("process-mode"):
        osd.set_property("process-mode", 1)

    # x86 建議設置 gpu-id 與關 QoS
    for e in (streammux, pgie, tracker, postconv, osd):
        if e and e.find_property("gpu-id"):
            e.set_property("gpu-id", gpu_id)
        if e and e.find_property("qos"):
            e.set_property("qos", 0)
        if e and e.find_property("nvbuf-memory-type"):
            e.set_property("nvbuf-memory-type", 0)

    # sink
    if is_aarch64():
        sink = (Gst.ElementFactory.make("nv3dsink", "sink")
                or Gst.ElementFactory.make("nveglglessink", "sink"))
    else:
        sink = (Gst.ElementFactory.make("nveglglessink", "sink")
                or Gst.ElementFactory.make("glimagesink", "sink")
                or Gst.ElementFactory.make("ximagesink", "sink")
                or Gst.ElementFactory.make("autovideosink", "sink"))
    if not sink:
        raise RuntimeError("Failed to create a video sink")
    sink.set_property("sync", sync)
    sink.set_property("async", False)

    # 加進 pipeline
    for e in (streammux, pgie, tracker, postconv, to_rgba, osd, sink):
        pipeline.add(e)

    # === 單一路 source bin，接到 mux.sink_0 ===
    srcbin = build_source_bin(0, uri)
    pipeline.add(srcbin)
    bin_srcpad = srcbin.get_static_pad("src")
    mux_sinkpad = streammux.get_request_pad("sink_0")
    if not mux_sinkpad:
        raise RuntimeError("Failed to get nvstreammux sink_0")
    if bin_srcpad.link(mux_sinkpad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link source bin → nvstreammux.sink_0")

    # === 連線：mux → pgie → tracker → postconv → RGBA → osd → sink ===
    assert streammux.link(pgie)
    assert pgie.link(tracker)
    assert tracker.link(postconv)
    assert postconv.link(to_rgba)
    assert to_rgba.link(osd)
    assert osd.link(sink)

    # === 掛 probe：在 tracker.src 做骨架/自訂 bbox ===
    # 確保已經把 tracker_src_pad_buffer_probe / set_custom_bbox / parse_pose_from_meta 與 pyds 匯入到此檔
    trk_src = tracker.get_static_pad("src")
    if trk_src:
        # 第三參數 user_data 可傳需要的物件，這裡用 None
        trk_src.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, None)

    # 保存參考，方便 finally 釋放
    pipeline._mux = streammux
    pipeline._mux_sinkpad = mux_sinkpad
    pipeline._srcbin = srcbin

    return pipeline


def parse_args():
    p = argparse.ArgumentParser(description="DeepStream minimal viewer")
    p.add_argument("-s", "--source", required=True, help="rtsp://... 或 /path/to/file")
    p.add_argument("-W", "--width",  type=int, default=1920)
    p.add_argument("-H", "--height", type=int, default=1080)
    p.add_argument("--sync", action="store_true", help="sink 與時鐘同步（預設不同步）")
    p.add_argument("-c", "--config-infer", default="config_infer_primary_yoloV8_pose.txt" ,help="nvinfer 的 config 檔 (.txt)")
    p.add_argument("-g", "--gpu-id", type=int, default=0, help="GPU id（x86 建議設）")
    return p.parse_args()


def main():
    # 建議：確保外掛路徑已設好（必要時自行 export）
    # os.environ.setdefault("GST_PLUGIN_PATH", "/opt/nvidia/deepstream/deepstream/lib/gst-plugins")
    # os.environ.setdefault("LD_LIBRARY_PATH", "/opt/nvidia/deepstream/deepstream/lib:" + os.environ.get("LD_LIBRARY_PATH",""))

    Gst.init(None)
    args = parse_args()

    # 統一把 file 路徑補成 URI
    src = args.source
    if src.startswith("/"):
        src = f"file://{os.path.abspath(src)}"

    pipeline = build_pipeline_deepstream(
        src, args.width, args.height,
        batch_size=1, sync=args.sync,
        config_infer=args.config_infer, gpu_id=args.gpu_id
    )

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    print("\n=== DeepStream Minimal Viewer ===")
    print(f"SOURCE        : {src}")
    print(f"MUX WxH       : {args.width}x{args.height}")
    print(f"SINK SYNC     : {args.sync}")
    print(f"JETSON        : {'TRUE' if is_aarch64() else 'FALSE'}\n")

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    sys.exit(main())
