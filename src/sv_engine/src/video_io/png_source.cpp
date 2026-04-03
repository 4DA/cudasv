#include <engine/png_source.hpp>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

namespace
{

void free_frames(std::array<AVFrame *, camera::CAMERAS_TOTAL> &frames)
{
    for (AVFrame *&frame: frames) {
        if (frame != nullptr) {
            av_frame_free(&frame);
        }
    }
}

void fail(const std::string &msg, int err)
{
    char buf[256];
    av_strerror(err, buf, sizeof(buf));
    std::cerr << "FAIL: " << msg << ": " << buf << "\n";
    std::exit(1);
}

AVFrame* convert_to_rgb24(const AVFrame *src)
{
    AVFrame *dst = av_frame_alloc();
    if (!dst) {
        fail("av_frame_alloc RGB dst", AVERROR(ENOMEM));
    }

    dst->format = AV_PIX_FMT_RGB24;
    dst->width = src->width;
    dst->height = src->height;

    int err = av_frame_get_buffer(dst, 32);
    if (err < 0) {
        fail("av_frame_get_buffer RGB dst", err);
    }

    SwsContext *sws = sws_getContext(src->width,
                                     src->height,
                                     static_cast<AVPixelFormat>(src->format),
                                     dst->width,
                                     dst->height,
                                     static_cast<AVPixelFormat>(dst->format),
                                     SWS_BILINEAR,
                                     nullptr,
                                     nullptr,
                                     nullptr);
    if (!sws) {
        fail("sws_getContext", AVERROR(EINVAL));
    }

    err = sws_scale(sws,
                    src->data,
                    src->linesize,
                    0,
                    src->height,
                    dst->data,
                    dst->linesize);
    if (err <= 0) {
        fail("sws_scale", AVERROR(EINVAL));
    }

    sws_freeContext(sws);
    return dst;
}

AVFrame* decode_png(const std::string &path)
{
    AVFormatContext *fmt = nullptr;

    const AVInputFormat *ifmt = av_find_input_format("image2");

    int err = avformat_open_input(&fmt, path.c_str(), ifmt, nullptr);
    if (err < 0) {
        fail("avformat_open_input " + path, err);
    }

    err = avformat_find_stream_info(fmt, nullptr);
    if (err < 0) {
        fail("avformat_find_stream_info", err);
    }

    const int video_stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream < 0) {
        fail("av_find_best_stream (video)", video_stream);
    }

    const AVStream *st = fmt->streams[video_stream];
    const AVCodec *dec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!dec) {
        fail("avcodec_find_decoder", AVERROR_DECODER_NOT_FOUND);
    }

    AVCodecContext *cc = avcodec_alloc_context3(dec);
    if (!cc) {
        fail("avcodec_alloc_context3", AVERROR(ENOMEM));
    }

    err = avcodec_parameters_to_context(cc, st->codecpar);
    if (err < 0) {
        fail("avcodec_parameters_to_context", err);
    }

    cc->thread_count = 0;

    err = avcodec_open2(cc, dec, nullptr);
    if (err < 0) {
        fail("avcodec_open2", err);
    }

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frm = av_frame_alloc();
    if (!pkt || !frm) {
        fail("alloc pkt/frm", AVERROR(ENOMEM));
    }

    AVFrame *rgb = nullptr;
    bool got_one = false;

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != video_stream) {
            av_packet_unref(pkt);
            continue;
        }

        err = avcodec_send_packet(cc, pkt);
        av_packet_unref(pkt);
        if (err < 0) {
            fail("avcodec_send_packet", err);
        }

        while (true) {
            err = avcodec_receive_frame(cc, frm);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
                break;
            }
            if (err < 0) {
                fail("avcodec_receive_frame", err);
            }

            if (!got_one) {
                rgb = convert_to_rgb24(frm);
                got_one = true;
            }
            av_frame_unref(frm);
        }

        if (got_one) {
            break;
        }
    }

    if (!got_one) {
        avcodec_send_packet(cc, nullptr);
        while (true) {
            err = avcodec_receive_frame(cc, frm);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
                break;
            }
            if (err < 0) {
                fail("receive_frame (flush)", err);
            }
            rgb = convert_to_rgb24(frm);
            got_one = true;
            av_frame_unref(frm);
            break;
        }
    }

    if (!got_one) {
        fail("No video frame decoded from " + path, AVERROR_INVALIDDATA);
    }

    av_frame_free(&frm);
    av_packet_free(&pkt);
    avcodec_free_context(&cc);
    avformat_close_input(&fmt);
    return rgb;
}

void pre_init()
{
    av_log_set_level(AV_LOG_INFO);
    avformat_network_init();
}

void fini()
{
    avformat_network_deinit();
}

} // namespace

namespace videoio
{

struct PNGSource::Impl
{
    std::array<AVFrame *, camera::CAMERAS_TOTAL> avFrames = {nullptr};
};

PNGSource::PNGSource():
    _impl(std::make_unique<Impl>())
{
}

PNGSource::~PNGSource()
{
    reset();
    fini();
}

bool PNGSource::start(std::array<std::string, camera::CAMERAS_TOTAL> sources)
{
    pre_init();
    reset();

    for (std::size_t index = 0; index < sources.size(); ++index) {
        _impl->avFrames[index] = decode_png(sources[index]);
    }

    return true;
}

void PNGSource::reset()
{
    free_frames(_impl->avFrames);
}

bool PNGSource::get_next_frame(FrameSet<camera::CAMERAS_TOTAL> &frames)
{
    frames.width = static_cast<uint32_t>(_impl->avFrames[0]->width);
    frames.height = static_cast<uint32_t>(_impl->avFrames[0]->height);
    frames.stride = static_cast<uint32_t>(_impl->avFrames[0]->linesize[0]);
    frames.timestamp = monotonic_ts_now();
    frames.frameseq = 0;

    for (std::size_t index = 0; index < _impl->avFrames.size(); ++index) {
        frames.data[index] = _impl->avFrames[index]->data[0];

        assert(frames.width == static_cast<uint32_t>(_impl->avFrames[index]->width));
        assert(frames.height == static_cast<uint32_t>(_impl->avFrames[index]->height));
    }

    return true;
}

bool PNGSource::release_frame(const FrameSet<camera::CAMERAS_TOTAL> &frame_set)
{
    (void)frame_set;
    return true;
}

} // namespace videoio
