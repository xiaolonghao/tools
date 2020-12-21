1.下载docker镜像（tf1.x为例）
docker pull nvcr.io/nvidia/tensorflow:20.12-tf1-py3
2.使用镜像实例化容器
nvidia-docker run -it -v hxl_docker:/docker_root --name hxl_tf1.15_python3.8  -p 1002:22    c8515d0efd50
3.初始化容器
apt update && apt install openssh-server -y
echo 'root:test' | chpasswd
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
echo "export VISIBLE=now" >> /etc/profile
service ssh restart
apt install zip unzip rar tmux   -y
mkdir  ~/.pip/
echo [global] > ~/.pip/pip.conf
echo index-url=https://pypi.tuna.tsinghua.edu.cn/simple >> ~/.pip/pip.conf
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/
conda config --set show_channel_urls yes
