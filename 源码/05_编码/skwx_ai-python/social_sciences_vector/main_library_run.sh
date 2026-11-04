#!/bin/bash

# 脚本名: run_api.sh

# Python服务文件名
PYTHON_FILE="main_library_api.py"
#PYTHON_ENV="llm_ai_development_py310"
LOG_FILE="log.txt"

## 1. 判断conda是否存在虚拟环境“python310_graph”
#if conda env list | grep -w $PYTHON_ENV; then
#    echo "激活虚拟环境 $PYTHON_ENV 已经存在。"
#    source activate $PYTHON_ENV
#else
#    # 2. 如果不存在，则创建虚拟环境
#    echo "创建虚拟环境 $PYTHON_ENV..."
#    conda create -n $PYTHON_ENV python=3.10 -y
#    # 3. 激活虚拟环境并根据当前目录下的requirements.txt文件安装包
#    echo "激活并安装依赖..."
#    source activate $PYTHON_ENV
#    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
#fi

# 2. 根据文件名查找进程号
PID=$(pgrep -f $PYTHON_FILE)

# 3. 检查是否找到进程
if [ -z "$PID" ]; then
    echo "没有找到运行中的$PYTHON_FILE进程。"
else
    # 杀死进程
    echo "正在关闭进程 $PID..."
    kill $PID
    echo "进程 $PID 已关闭。"
fi
# 等待进程完全关闭
sleep 5

# 4. 启动python脚本，并后台运行，输出日志到log.txt
nohup python $PYTHON_FILE &> $LOG_FILE &

# 输出提示信息
echo "Python script is running in the background. Logs are being written to log.txt."

# 5. 监听log.txt文件输出日志
tail -f $LOG_FILE

