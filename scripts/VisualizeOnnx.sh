#!/bin/bash

# 默认测试基路径
BASE_PATH="/app/Deeploy/DeeployTest/Tests"

# 检查是否传递了测试名称参数
if [ -z "$1" ]; then
    echo "❌ Please provide a test name as an argument. Example: ./visualizeCCTonnx.sh testFloatGemm"
    exit 1
fi

# 根据传入的测试名称构建 ONNX 文件路径
TEST_NAME=$1
ONNX_PATH="${BASE_PATH}/${TEST_NAME}/network.onnx"

# 检查 ONNX 文件是否存在
if [ ! -f "$ONNX_PATH" ]; then
    echo "❌ The ONNX file does not exist: $ONNX_PATH"
    exit 1
fi

# 自动查找未被占用的端口
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 检查 Netron 是否已安装
if ! command -v netron &> /dev/null; then
    echo "❌ Netron is not installed. Installing now..."
    pip install netron
fi

# 在后台运行 Netron
echo "🚀 Starting Netron in the background on http://localhost:$PORT ..."
nohup netron -p $PORT "$ONNX_PATH" > /dev/null 2>&1 &

# 打印访问地址
echo "✅ Netron is running in the background."
echo "🌐 Access it at: http://localhost:$PORT"
