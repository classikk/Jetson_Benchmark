#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <cstring>

struct Buffer {
    void* start;
    size_t length;
};

int main() {
    const char* dev_name = "/dev/video0";
    int fd = open(dev_name, O_RDWR);
    if (fd < 0) {
        perror("Cannot open device");
        return 1;
    }
    #define V4L2_CID_SENSOR_MODE (0x009a2008);
    // --- Set controls ---
    v4l2_control ctrl;
    ctrl.id = V4L2_CID_SENSOR_MODE; // sensor_mode
    ctrl.value = 4;
    if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) perror("Setting sensor_mode failed");
//
    //#define V4L2_CID_BYPASS_MODE (0x009a2064);
    //ctrl.id = V4L2_CID_BYPASS_MODE; // bypass_mode
    //ctrl.value = 0;
    //if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) perror("Setting bypass_mode failed");

    // --- Set format ---
    v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    constexpr int width  = 1280;
    constexpr int height = 720;
    fmt.fmt.pix.width  = width ;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = v4l2_fourcc('R','G','1','0'); // RG10
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("Setting format failed");
        return 1;
    }

    // --- Request buffers ---
    v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("Requesting buffer failed");
        return 1;
    }

    Buffer* buffers = new Buffer[req.count];
    for (__u32 i = 0; i < req.count; i++) {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("Query buffer failed");
            return 1;
        }

        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED) {
            perror("mmap failed");
            return 1;
        }
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) perror("Queue buffer failed");
    }

    // --- Start streaming ---
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        perror("Stream on failed");
        return 1;
    }

    std::ofstream outfile("FILE.raw", std::ios::binary);

    // --- Capture 120 frames ---
    for (int frame = 0; frame < 120; frame++) {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
            perror("Dequeue buffer failed");
            continue;
        }
        int image_size = 3*height*width;
        char* result = new char[image_size];
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                int pixel = y*width+x;
                char* write_to = result + 3*pixel;
                write_to[0] = (((short*)buffers[buf.index].start)[pixel]);
                write_to[1] = (((short*)buffers[buf.index].start)[pixel]);
                write_to[2] = (((short*)buffers[buf.index].start)[pixel]);
            }
        }

        outfile.write(result, image_size);
        //outfile.write((char*)buffers[buf.index].start, buf.bytesused);
        delete[] result;
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) perror("Requeue buffer failed");
    }

    outfile.close();

    // --- Stop streaming ---
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0) perror("Stream off failed");

    // --- Cleanup ---
    for (__u32 i = 0; i < req.count; i++) {
        munmap(buffers[i].start, buffers[i].length);
    }
    delete[] buffers;
    close(fd);

    return 0;
}
