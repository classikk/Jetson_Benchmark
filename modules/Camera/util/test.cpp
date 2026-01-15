#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class DummyImagePublisher : public rclcpp::Node
{
public:
  DummyImagePublisher()
  : Node("dummy_image_publisher")
  {
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&DummyImagePublisher::publish_image, this));
  }

private:
  void publish_image()
  {
    // Image parameters
    const uint32_t width = 320;
    const uint32_t height = 240;
    const std::string encoding = "rgb8";  // 3 channels, 8-bit each

    sensor_msgs::msg::Image img_msg;
    img_msg.header.stamp = this->now();
    img_msg.header.frame_id = "camera_frame";
    img_msg.height = height;
    img_msg.width = width;
    img_msg.encoding = encoding;
    img_msg.is_bigendian = false;
    img_msg.step = width * 3;  // 3 bytes per pixel for rgb8

    // Fill with dummy pattern data
    img_msg.data.resize(img_msg.step * img_msg.height);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        size_t idx = y * img_msg.step + x * 3;
        img_msg.data[idx + 0] = static_cast<uint8_t>(x % 256);       // R
        img_msg.data[idx + 1] = static_cast<uint8_t>(y % 256);       // G
        img_msg.data[idx + 2] = static_cast<uint8_t>((x + y) % 256); // B
      }
    }

    publisher_->publish(img_msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DummyImagePublisher>());
  rclcpp::shutdown();
  return 0;
}
