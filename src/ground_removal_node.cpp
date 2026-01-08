#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

namespace harrp {

class GroundRemovalNode : public rclcpp::Node
{
public:
  explicit GroundRemovalNode(const rclcpp::NodeOptions & options)
  : Node("ground_removal_node", options)
  {
    // 1. パラメータの宣言
    // RoboSense論文の検出範囲 z=[-1.0, 4.0] を参考に、初期値を設定します [cite: 221]
    this->declare_parameter("z_threshold", -0.12); 
    z_threshold_ = this->get_parameter("z_threshold").as_double();

    // 2. パラメータ変更イベントの登録
    // このハンドルを保持し続ける必要があります
    callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&GroundRemovalNode::on_set_parameters, this, std::placeholders::_1));

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.best_effort();
    qos.durability_volatile();

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar", qos,
      std::bind(&GroundRemovalNode::topic_callback, this, std::placeholders::_1));

    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar/no_ground", 10);

    RCLCPP_INFO(this->get_logger(), "GroundRemovalNode initialized with z_threshold: %f", z_threshold_);
  }

private:
  // パラメータが変更された時に呼ばれるコールバック
  rcl_interfaces::msg::SetParametersResult on_set_parameters(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    auto result = rcl_interfaces::msg::SetParametersResult();
    result.successful = true;

    for (const auto & param : parameters) {
      if (param.get_name() == "z_threshold") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
          z_threshold_ = param.as_double();
          RCLCPP_INFO(this->get_logger(), "z_threshold updated to: %f", z_threshold_);
        } else {
          result.successful = false;
          result.reason = "z_threshold must be a double";
        }
      }
    }
    return result;
  }

  void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    int z_offset = -1;
    for (const auto& field : msg->fields) {
      if (field.name == "z") {
        z_offset = field.offset;
        break;
      }
    }
    if (z_offset == -1) return;

    auto output_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    output_msg->header = msg->header;
    output_msg->height = 1;
    output_msg->fields = msg->fields;
    output_msg->is_bigendian = msg->is_bigendian;
    output_msg->point_step = msg->point_step;
    output_msg->is_dense = msg->is_dense;

    output_msg->data.resize(msg->data.size());

    const uint8_t* raw_input_ptr = msg->data.data();
    uint8_t* raw_output_ptr = output_msg->data.data();
    
    size_t point_step = msg->point_step;
    size_t num_points = msg->width * msg->height;
    size_t current_output_offset = 0;
    size_t valid_points_count = 0;

    // 現在のしきい値を使用してフィルタリング
    double current_threshold = z_threshold_; 

    for (size_t i = 0; i < num_points; ++i) {
      const uint8_t* current_point_ptr = raw_input_ptr + (i * point_step);
      float z_value;
      std::memcpy(&z_value, current_point_ptr + z_offset, sizeof(float));

      if (z_value > current_threshold) {
        std::memcpy(raw_output_ptr + current_output_offset, current_point_ptr, point_step);
        current_output_offset += point_step;
        valid_points_count++;
      }
    }

    output_msg->data.resize(current_output_offset);
    output_msg->width = valid_points_count;
    output_msg->row_step = output_msg->point_step * output_msg->width;

    pub_->publish(std::move(output_msg));
  }

  double z_threshold_;
  // パラメータコールバックのハンドル
  OnSetParametersCallbackHandle::SharedPtr callback_handle_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

} // namespace harrp

// main関数は変更なし
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<harrp::GroundRemovalNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
