#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp> // 追加
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

// namespaceをつけると管理しやすいです
namespace harrp {

class GroundRemovalNode : public rclcpp::Node
{
public:
  // コンストラクタの引数に NodeOptions を追加 (コンポーネント化に必須)
  explicit GroundRemovalNode(const rclcpp::NodeOptions & options)
  : Node("ground_removal_node", options)
  {
    // ... (中身は前回の最速版と同じ) ...
    // パラメータ宣言
    this->declare_parameter("z_threshold", -0.13);
    z_threshold_ = this->get_parameter("z_threshold").as_double();

    // QoS設定 (IPCを有効にするために重要)
    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.best_effort();
    qos.durability_volatile();

    // サブスクライバー
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar", qos,
      std::bind(&GroundRemovalNode::topic_callback, this, std::placeholders::_1));

    // パブリッシャー
    // Intra-process Communicationを有効にするには、パブリッシャー作成時にオプション等は不要ですが
    // 相手も同じコンテナにいる必要があります。
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar/no_ground", 10);
      
    // 推奨: パブリッシュ時に一意のポインタを使うとゼロコピーが確実になりますが、
    // unique_ptrで渡しているので基本的にはOKです。
  }

private:
  void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // ... (前回の最速版と同じ処理) ...
    // 変更なし
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

    for (size_t i = 0; i < num_points; ++i) {
      const uint8_t* current_point_ptr = raw_input_ptr + (i * point_step);
      float z_value;
      std::memcpy(&z_value, current_point_ptr + z_offset, sizeof(float));

      if (z_value > z_threshold_) {
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
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

} // namespace harrp


// main関数を削除し、コンポーネント登録マクロを追加
RCLCPP_COMPONENTS_REGISTER_NODE(harrp::GroundRemovalNode)
