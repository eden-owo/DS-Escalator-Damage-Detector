import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import os
import sys
import time
import argparse
import platform

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
Gst.init(None)

import pyds

SOURCE = ''
CONFIG_INFER = ''
STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 720
STREAMMUX_HEIGHT = 1280
GPU_ID = 0
PERF_MEASUREMENT_INTERVAL_SEC = 5

start_time = time.time()
fps_streams = {}

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


def tracker_src_pad_buffer_probe(pad, info, user_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except Exception:
            break

        stream_id = frame_meta.source_id  # 0,1,2...
        key = f'stream{stream_id}'
        if key in fps_streams:
            fps_streams[key].get_fps()

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


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

    pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
    if not pgie:
        sys.stderr.write('ERROR: Failed to create nvinfer\n')
        sys.exit(1)

    tracker = Gst.ElementFactory.make('nvtracker', 'nvtracker')
    if not tracker:
        sys.stderr.write('ERROR: Failed to create nvtracker\n')
        sys.exit(1)

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




    sys.stdout.write('\n')
    sys.stdout.write('SOURCE: %s\n' % SOURCE)
    sys.stdout.write('CONFIG_INFER: %s\n' % CONFIG_INFER)
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
    pgie.set_property('config-file-path', CONFIG_INFER)
    pgie.set_property('qos', 0)
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

    if not is_aarch64():
        streammux.set_property('nvbuf-memory-type', 0)
        streammux.set_property('gpu_id', GPU_ID)
        pgie.set_property('gpu_id', GPU_ID)
        tracker.set_property('gpu_id', GPU_ID)
        converter.set_property('nvbuf-memory-type', 0)
        converter.set_property('gpu_id', GPU_ID)
        osd.set_property('gpu_id', GPU_ID)

    # 加入 pipeline
    for elem in [streammux, source_bin, pgie, tracker, 
                tee, queue_osd, converter, osd, sink]:
        pipeline.add(elem)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(tee)

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


    # 監聽 pipeline 的 bus，處理錯誤/結束訊息    
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', bus_call, loop)

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
    global SOURCE, CONFIG_INFER, STREAMMUX_BATCH_SIZE, STREAMMUX_WIDTH, STREAMMUX_HEIGHT, GPU_ID, \
        PERF_MEASUREMENT_INTERVAL_SEC

    parser = argparse.ArgumentParser(description='DeepStream')
    parser.add_argument('-s', '--source', required=True, help='Source stream/file')
    parser.add_argument('-c', '--config-infer', required=True, help='Config infer file')
    parser.add_argument('-b', '--streammux-batch-size', type=int, default=1, help='Streammux batch-size (default: 1)')
    parser.add_argument('-w', '--streammux-width', type=int, default=720, help='Streammux width (default: 1920)')
    parser.add_argument('-e', '--streammux-height', type=int, default=1280, help='Streammux height (default: 1080)')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, help='GPU id (default: 0)')
    parser.add_argument('-f', '--fps-interval', type=int, default=5, help='FPS measurement interval (default: 5)')
    args = parser.parse_args()
    if args.source == '':
        sys.stderr.write('ERROR: Source not found\n')
        sys.exit(1)
    if args.config_infer == '' or not os.path.isfile(args.config_infer):
        sys.stderr.write('ERROR: Config infer not found\n')
        sys.exit(1)

    SOURCE = args.source
    CONFIG_INFER = args.config_infer
    STREAMMUX_BATCH_SIZE = args.streammux_batch_size
    STREAMMUX_WIDTH = args.streammux_width
    STREAMMUX_HEIGHT = args.streammux_height
    GPU_ID = args.gpu_id
    PERF_MEASUREMENT_INTERVAL_SEC = args.fps_interval


if __name__ == '__main__':
    parse_args()
    sys.exit(main())