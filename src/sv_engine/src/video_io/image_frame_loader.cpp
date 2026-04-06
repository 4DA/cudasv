#include <cstdlib>
#include <memory>
#include <string>

#include <engine/image_frame_loader.hpp>

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

namespace
{

static std::string ffmpeg_error_string(int err)
{
    char buffer[256];
    av_strerror(err, buffer, sizeof(buffer));
    return buffer;
}

static bool convert_to_rgb24(const AVFrame *src,
                             videoio::DecodedImageFrame &frame,
                             std::string *errorMessage)
{
    AVFrame *rgbFrame = av_frame_alloc();
    if (!rgbFrame) {
        if (errorMessage) {
            *errorMessage = "av_frame_alloc failed";
        }
        return false;
    }

    rgbFrame->format = AV_PIX_FMT_RGB24;
    rgbFrame->width = src->width;
    rgbFrame->height = src->height;

    int err = av_frame_get_buffer(rgbFrame, 32);
    if (err < 0) {
        if (errorMessage) {
            *errorMessage = "av_frame_get_buffer failed: " + ffmpeg_error_string(err);
        }
        av_frame_free(&rgbFrame);
        return false;
    }

    SwsContext *scaleContext = sws_getContext(src->width,
                                              src->height,
                                              static_cast<AVPixelFormat>(src->format),
                                              rgbFrame->width,
                                              rgbFrame->height,
                                              static_cast<AVPixelFormat>(rgbFrame->format),
                                              SWS_BILINEAR,
                                              nullptr,
                                              nullptr,
                                              nullptr);
    if (!scaleContext) {
        if (errorMessage) {
            *errorMessage = "sws_getContext failed";
        }
        av_frame_free(&rgbFrame);
        return false;
    }

    err = sws_scale(scaleContext,
                    src->data,
                    src->linesize,
                    0,
                    src->height,
                    rgbFrame->data,
                    rgbFrame->linesize);
    sws_freeContext(scaleContext);
    if (err <= 0) {
        if (errorMessage) {
            *errorMessage = "sws_scale failed";
        }
        av_frame_free(&rgbFrame);
        return false;
    }

    frame.owner = std::shared_ptr<void>(rgbFrame, [](void *pointer) {
        AVFrame *ownedFrame = static_cast<AVFrame *>(pointer);
        av_frame_free(&ownedFrame);
    });
    frame.data = rgbFrame->data[0];
    frame.width = static_cast<uint32_t>(rgbFrame->width);
    frame.height = static_cast<uint32_t>(rgbFrame->height);
    frame.stride = static_cast<uint32_t>(rgbFrame->linesize[0]);
    return true;
}

} // namespace

namespace videoio
{

bool load_rgb_image(const std::string &path,
                    DecodedImageFrame &frame,
                    std::string *errorMessage)
{
    frame = {};

    avformat_network_init();

    AVFormatContext *formatContext = nullptr;
    const AVInputFormat *inputFormat = av_find_input_format("image2");
    int err = avformat_open_input(&formatContext, path.c_str(), inputFormat, nullptr);
    if (err < 0) {
        if (errorMessage) {
            *errorMessage = "avformat_open_input failed for '" + path + "': " + ffmpeg_error_string(err);
        }
        return false;
    }

    err = avformat_find_stream_info(formatContext, nullptr);
    if (err < 0) {
        if (errorMessage) {
            *errorMessage = "avformat_find_stream_info failed: " + ffmpeg_error_string(err);
        }
        avformat_close_input(&formatContext);
        return false;
    }

    const int videoStream = av_find_best_stream(formatContext,
                                                AVMEDIA_TYPE_VIDEO,
                                                -1,
                                                -1,
                                                nullptr,
                                                0);
    if (videoStream < 0) {
        if (errorMessage) {
            *errorMessage = "av_find_best_stream failed: " + ffmpeg_error_string(videoStream);
        }
        avformat_close_input(&formatContext);
        return false;
    }

    const AVStream *stream = formatContext->streams[videoStream];
    const AVCodec *decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!decoder) {
        if (errorMessage) {
            *errorMessage = "avcodec_find_decoder failed";
        }
        avformat_close_input(&formatContext);
        return false;
    }

    AVCodecContext *codecContext = avcodec_alloc_context3(decoder);
    if (!codecContext) {
        if (errorMessage) {
            *errorMessage = "avcodec_alloc_context3 failed";
        }
        avformat_close_input(&formatContext);
        return false;
    }

    err = avcodec_parameters_to_context(codecContext, stream->codecpar);
    if (err < 0) {
        if (errorMessage) {
            *errorMessage = "avcodec_parameters_to_context failed: " + ffmpeg_error_string(err);
        }
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return false;
    }

    err = avcodec_open2(codecContext, decoder, nullptr);
    if (err < 0) {
        if (errorMessage) {
            *errorMessage = "avcodec_open2 failed: " + ffmpeg_error_string(err);
        }
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return false;
    }

    AVPacket *packet = av_packet_alloc();
    AVFrame *decodedFrame = av_frame_alloc();
    if (!packet || !decodedFrame) {
        if (errorMessage) {
            *errorMessage = "failed to allocate packet/frame";
        }
        av_packet_free(&packet);
        av_frame_free(&decodedFrame);
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return false;
    }

    bool decoded = false;

    while (av_read_frame(formatContext, packet) >= 0 && !decoded) {
        if (packet->stream_index != videoStream) {
            av_packet_unref(packet);
            continue;
        }

        err = avcodec_send_packet(codecContext, packet);
        av_packet_unref(packet);
        if (err < 0) {
            if (errorMessage) {
                *errorMessage = "avcodec_send_packet failed: " + ffmpeg_error_string(err);
            }
            break;
        }

        while (true) {
            err = avcodec_receive_frame(codecContext, decodedFrame);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
                break;
            }
            if (err < 0) {
                if (errorMessage) {
                    *errorMessage = "avcodec_receive_frame failed: " + ffmpeg_error_string(err);
                }
                break;
            }

            decoded = convert_to_rgb24(decodedFrame, frame, errorMessage);
            av_frame_unref(decodedFrame);
            break;
        }
    }

    if (!decoded && err >= 0) {
        avcodec_send_packet(codecContext, nullptr);
        while (true) {
            err = avcodec_receive_frame(codecContext, decodedFrame);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
                break;
            }
            if (err < 0) {
                if (errorMessage) {
                    *errorMessage = "avcodec_receive_frame (flush) failed: " + ffmpeg_error_string(err);
                }
                break;
            }

            decoded = convert_to_rgb24(decodedFrame, frame, errorMessage);
            av_frame_unref(decodedFrame);
            break;
        }
    }

    av_frame_free(&decodedFrame);
    av_packet_free(&packet);
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);

    if (!decoded && errorMessage && errorMessage->empty()) {
        *errorMessage = "no decodable image frame found";
    }

    return decoded;
}

} // namespace videoio
