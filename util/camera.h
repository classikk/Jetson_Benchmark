#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <cstring>
#include "image.h"


struct Buffer {
    void* start;
    size_t length;
};

struct Streamer {
    const char* dev_name = "/dev/video0";
    int fd = open(dev_name, O_RDWR);
    // --- Set format ---
    v4l2_format fmt;
    int n_buffers = 1;
    Buffer* buffers = new Buffer[n_buffers];  
    const int width  = 1280;
    const int height = 720;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    bool init(){
        if (fd < 0) {
            perror("Cannot open device");
            return false;
        }

        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width  = width ;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.pixelformat = v4l2_fourcc('R','G','1','0'); // RG10
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
            perror("Setting format failed");
            return false;
        }
        // --- Request buffers ---
        v4l2_requestbuffers req;
        memset(&req, 0, sizeof(req));
        req.count = n_buffers;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
          
        if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
            perror("Requesting buffer failed");
            return false;
        }
        for (__u32 i = 0; i < n_buffers; i++) {
            v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;

            if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
                perror("Query buffer failed");
                return false;
            }

            buffers[i].length = buf.length;
            buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, buf.m.offset);
            if (buffers[i].start == MAP_FAILED) {
                perror("mmap failed");
            }
            if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
                perror("Queue buffer failed");
                return false;
            }
        }
        // --- Start streaming ---
        if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
            perror("Stream on failed");
            return false;
        }
        return true;
    }

    RG10 get_frame() {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
            perror("Dequeue buffer failed");
        }
        RG10 rawImage((char*)buffers[buf.index].start,width,height);
        return rawImage;
    }
    
    void record_new_image() {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) perror("Requeue buffer failed");
    }

    void cleanUp() {
        // --- Stop streaming ---
        if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0) perror("Stream off failed");

        // --- Cleanup ---
        for (__u32 i = 0; i < n_buffers; i++) {
            munmap(buffers[i].start, buffers[i].length);
        }
        delete[] buffers;
        close(fd);
    }
};
