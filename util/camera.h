#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <cstring>
#include "image.h"

class Streamer {
private:
    const char* dev_name = "/dev/video0";
    const int width  = 1280;
    const int height = 720;
    int n_buffers = 2;
    int fd;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    v4l2_format fmt;
    RG10* buffers = new RG10[n_buffers]; 
    bool stream_is_on = false;

public:
    bool init(){
        if (init_steps()){
            return true;
        }
        cleanUp();
        return false;
    }
    bool init_steps(){
        if (!init_device()) return false;
        if (!init_fmt()) return false;
        if (!init_request()) return false;
        set_image_info_RG10();
        if (!init_map_CPU()) return false;
        if (!init_start_stream()) return false;
        return true;
    }

    RG10 get_frame() {
        v4l2_buffer buf = get_v4l2_buffer();

        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
            perror("Dequeue buffer failed");
        }
        return buffers[buf.index];
    }
    
    void record_new_image() {
        v4l2_buffer buf = get_v4l2_buffer();
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) perror("Requeue buffer failed");
    }

    void cleanUp() {
        // --- Stop streaming ---
        if (stream_is_on){
            if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0) perror("Stream off failed");
        }
        
        // --- Cleanup ---
        for (__u32 i = 0; i < n_buffers; i++) {
            munmap(buffers[i].start, buffers[i].info.size());
        }
        delete[] buffers;
        close(fd);
    }

private:
    bool init_device(){
        fd = open(dev_name, O_RDWR);
        if (fd < 0) {
            perror("Cannot open device");
            return false;
        }
        return true;
    }

    bool init_fmt(){
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
        return true;
    }
    bool init_request(){
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
        return true;
    }
    void set_image_info_RG10(){
        IMG_info info(width,height,RG10::pix_width);
        for (__u32 i = 0; i < n_buffers; i++) {
            buffers[i].info = info;
        }
    }

    bool init_map_CPU(){
        for (__u32 i = 0; i < n_buffers; i++) {
            buffers[i].info.Is_GPU_pointer = false;
            if (!init_map_CPU_buf(i)) return false;
        }
        return true;
    }
    bool init_map_CPU_buf(int i){
        v4l2_buffer buf = get_v4l2_buffer();
        buf.index = i;

        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("Query buffer failed");
            return false;
        }

        if (buffers[i].info.size() != buf.length) {
            perror("Incorrect image size");
            cout << "Expecting total of bytes = " << buffers[i].info.size() << endl;
            cout << "Recieved  total of bytes = " << buf.length             << endl;
            return false;
        }

        buffers[i].start = (char*)mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED) {
            perror("mmap failed");
            return false;
        }
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("Queue buffer failed");
            return false;
        }
        return true;
    }
    bool init_start_stream(){
        // --- Start streaming ---
        if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
            perror("Stream on failed");
            return false;
        }
        stream_is_on = true;
        return true;
    }
    v4l2_buffer get_v4l2_buffer() {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        return buf;
    }
};
