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

# Installation
HARRPは、以下2つのリポジトリ (図上部2つのDockerコンテナ) のインストールが完了すれば使用できます。

* [docker_ros2_tao_pintpillars](https://github.com/HappyYusuke/docker_ros2_tao_pointpillars.git)
* [docker_ReID3D2025](https://github.com/HappyYusuke/docker_ReID3D2025.git)

</br>

<p align="center">
  <img src=figs/Software_stack.png width=500>
</p>

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

</br>

# Usage

