# HARRP
3D-LiDARを用いた再識別可能な人追従のROS2パッケージです。 <br>
本リポジトリでは、以下のリポジトリを使用することで、再識別機能を有した人追従を実現しています。

* [ros2_tao_pointpillars](https://github.com/NVIDIA-AI-IOT/ros2_tao_pointpillars.git)
* [ReID3D](https://github.com/GWxuan/ReID3D.git)

また、センサはLivox社の[MID-360](https://www.livoxtech.com/mid-360)で、[livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2.git)を使用しました。

</br>

<p align="center">
  <img src=figs/graphical_abst.png width=800>
</p>

### Demonstration Video
👉 [【Demonstration Video】HARRP: Human-following Autonomous Robot System Using ReID3D and PointPillars.](https://youtu.be/31dsITqNGis)

</br>

# Installation
HARRPは、以下2つのリポジトリ (図上部2つのDockerコンテナ) のインストールが完了すれば使用できます。

* [docker_ros2_tao_pintpillars](https://github.com/HappyYusuke/docker_ros2_tao_pointpillars.git)
* [docker_ReID3D2025](https://github.com/HappyYusuke/docker_ReID3D2025.git)

</br>

<p align="center">
  <img src=figs/Software_stack.png width=500>
</p>

</br>

## docker_ros2_tao_pintpillars
リポジトリをクローンする

```bash
git clone https://github.com/HappyYusuke/docker_ros2_tao_pointpillars.git
```

</br>

HARRP向けに学習された重みを以下からダウンロードする。</br>
https://kanazawa-it.box.com/s/lxcm43tq1e1so6y4po3pkop96rxc5640

Dockerコンテナのホームディレクトリに移動
```bash
mv ~/Downloads/harrp_weight.onnx ~/docker_ros2_tao_pointpillars/home
```

</br>

> [!NOTE]
> Dockerがインストールされていない場合
> ```bash
> 本リポジトリに移動
> cd ~/docker_ros2_tao_pointpillars
>
> # Dockerをインストール
> ./install-docker.sh
> ```

</br>

Dockerを起動する。<br>
Docker Imageのロードが始まり、コンテナが起動するとプロンプトの@以降がros2になる。

```
./run-docker-container.sh
```

</br>

セットアップ用のシェルスクリプトを実行してください。

```bash
./setup.sh
```

</br>

## docker_ReID3D2025
リポジトリをクローンする。

```
git clone https://github.com/HappyYusuke/docker_ReID3D2025.git
```

</br>

Dockerを起動する。<br>
Docker Imageのロードが始まり、起動するとプロンプトの@以降がros2になる。

```
./run-docker-containter.sh
```

</br>

`setup.sh`を実行することでセットアップが完了します。

```
./setup.sh
```

<br>

zipファイルを以下URLからダウンロードする。</br>
https://kanazawa-it.box.com/s/jsde13gu1vscmgggf073i9a3vtfh0xob

<br>

ホストPCに戻ります。<br>
ダウンロードしたzipファイルを解凍し移動する。
```
# 解凍
cd ~/Downloads
unzip large_files_docker_ReID3D2025.zip

# 重みファイルを移動
mv large_files_docker_ReID3D2025/ckpt_best.pth ~/docker_ReID3D2025/home/ReID3D/reidnet/log
```

</br>

# Usage
MID-360を初めて接続する場合は以下の「MID-360と接続したい場合」を行ってください。

<details>
<summary>MID-360と接続したい場合</summary>

### イーサネットを設定します。
`/etc/netplan/`内に以下の内容のファイルを任意のファイル名で作成してください。拡張子は`.yaml`です。</br>
また、元からあるファイルは別の場所に移動してください。

```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    # 下の "enXXXXXXXXXXXX" をPCに合わせて変更してください (コマンド「ip a」で、enから始まるポート名) 。
    enXXXXXXXXXXXX:
      dhcp4: false
      addresses: [192.168.1.50/24]  # Livoxと通信するためのPC側のIP
      optional: true
```

</br>

ファイルを元に設定を反映します。
```bash
sudo netplan apply
```

</br>

### `livox_ros_driver2`の設定ファイルを書き換えます。

1. `./run-docker-containter.sh`でDockerを起動します。
   
2. `MID360_config.json`を開きます。
```
vim ~/colcon_ws/src/livox_ros_driver2/config/MID360_config.json
```
3. `host_net_info`内のipを`192.168.1.50`に変更します。具体的な変更箇所は以下の通りです。

    - `"cmd_data_ip" : "192.168.1.50",`
    - `"push_msg_ip": "192.168.1.50",`
    - `"point_data_ip": "192.168.1.50",`
    - `"imu_data_ip" : "192.168.1.50",`

4. `lidar_configs`のipを以下の手順で変更します。

    - お手元のMID-360のシリアル番号末尾2桁をご確認ください（ここでは例として`15`とします）。
    - MID-360は`192.168.1.1XX/24`のいずれかに設定されます。（`192.168.1.115`となります）。
    - `ping 192.168.1.1XX`を実行し、応答があることを確認します。
    - 応答が確認できたら、`lidar_configs`のipアドレスを変更してください。

<br>

### `launch_ROS2/msg_MID360_launch.py`のパラメータを変更します。
launchファイルを開きます。
```bash
vim ~/colcon_ws/src/livox_ros_driver2/launch_ROS2/msg_MID360_launch.py
```

</br>

`xfer_format   = 1`を`xfer_format   = 0`にしてください。

</br>

コンテナ内でビルド
```bash
cd ~/colcon_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source ~/colcon_ws/install/setup.bash
```

</details>

</br>

## ros2_tao_pointpillarsを実行
`docker_ros2_tao_pointpillars`のコンテナ内で以下を実行。
```bash
ros2 launch pp_infer pp_infer_harrp_launch.py
```

</br>

## HARRPを実行
`docker_ReID3D2025`のコンテナ内で`terminator`を起動してください。
```bash
terminator
```

`terminator`は以下の通りターミナルを分割できます。
- ctrl+shift+oで上下分割
- ctrl+shift+eで左右分割
- ctrl+shift+nや+pで画面間移動
- ctrl+shift+wで画面を一つ閉じる

</br>

以下コマンドを実行していき、ロボットの正面に人が立つと追従対象が登録され、追従が始まります。
```bash
# rviz2
rviz2 -d ~/colcon_ws/src/harrp/rviz/harrp_kachaka.rviz

# livox_ros_driver2
ros2 launch livox_ros_driver2 msg_MID360_launch.py

# HARRP
ros2 launch harrp rviz_harrp_launch.py
```

</br>

# Author
金澤 祐典 (金沢工業大学大学院 工学研究科 機械工学専攻 出村研究室)
