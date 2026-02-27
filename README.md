# HARRP
3D-LiDARを用いた再識別可能な人追従のROS2パッケージです。 <br>
本リポジトリでは、以下のリポジトリを使用することで、再識別機能を有した人追従を実現しています。

* [ros2_tao_pointpillars](https://github.com/NVIDIA-AI-IOT/ros2_tao_pointpillars.git)
* [ReID3D](https://github.com/GWxuan/ReID3D.git)

また、センサはLivox社の[MID-360](https://www.livoxtech.com/mid-360)で、[livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2.git)を使用しました。

</br>

<img src=figs/graphical_abst.png width=800>

### Demonstration Video
👉 [【Demonstration Video】HARRP: Human-following Autonomous Robot System Using ReID3D and PointPillars.](https://youtu.be/31dsITqNGis)

# Installation
HARRPは、以下2つのリポジトリのインストールが完了すれば使用できます。

* [docker_ros2_tao_pintpillars](https://github.com/HappyYusuke/docker_ros2_tao_pointpillars.git)
* [docker_ReID3D2025](https://github.com/HappyYusuke/docker_ReID3D2025.git)

## docker_ros2_tao_pintpillars
リポジトリをクローンする

```bash
git clone https://github.com/HappyYusuke/docker_ros2_tao_pointpillars.git
```

</br>

HARRP向けに学習された重みを使用する場合は以下からダウンロードする。</br>
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

> [!TIP]
> 金沢工業大学のプロキシ環境下で開発する場合
> ```bash
> # プロキシを設定する
> setkitproxy
>
> # プロキシを設定しない
> unkitproxy
> ```

以下3つのリポジトリを使用するため、セットアップ用のシェルスクリプトを実行してください。
* [ros2_tao_pointpillars](https://github.com/HappyYusuke/ros2_tao_pointpillars.git) (HARRP用に調整)
* [harrp](https://github.com/HappyYusuke/harrp.git)
* [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2.git)

```bash
./setup.sh
```

## 
