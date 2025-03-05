# SmolVLM-256M-Instruct 模型转换
这个模型分为 Vision Encoder 和 Language Model 两部分，分别进行转换。

## 一、转换 Vision Encoder 

导出 Vision Encoder 为 onnx，然后通过 `pulsar2 build` 转换为 axmodel模型，

### 1. 创建虚拟环境

```
conda create -n smolvlm python=3.12 -y
conda activate smolvlm
```

### 2. 安装依赖

```
pip install -r requirements.txt
```
本代码依赖 `transformers==4.49.0`，最好使用这个版本的`transformers`，其它版本可能会有不兼容问题。

### 3. 导出模型（PyTorch -> ONNX）

在导出onnx之前需要先下从 huggingface 或 model scope 下载模型。这里假设模型的保存目录是 `../SmolVLM-256M-Instruct/`。    

可以执行`bash export.sh`直接导出模型，以下是详细步骤。  

1). 运行模型，看一下float模型的输出
```
python run.py ../SmolVLM-256M-Instruct/
```

2). 导出onnx模型
```
python export.py ../SmolVLM-256M-Instruct/
```
这一步会生成 `Qwen2.5-VL-3B-Instruct_vision.onnx`和`Qwen2.5-VL-3B-Instruct_vision.onnx.data`,计算图和权重参数分离。

3). 测试onnx模型

```
python test_onnx.py ../SmolVLM-256M-Instruct/
```
这一步会用onnx模型替换 vision encoder 模块进行推理，可以对比一下 `run.py`的输出是否一致。

### 4.转换模型（ONNX -> Axera）

使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

1). 量化数据集  
下载现有数据集[imagenet-calib.tar](https://github.com/AXERA-TECH/SmolVLM-256M-Instruct.axera/releases/download/v1.0.0/imagenet-calib.tar) 或者自己tar打包图片数据。
```
tar  -cvf calib.tar {your images}
```

2). 模型转换

* 修改配置文件
 
检查`config.json` 中 `calibration_dataset` 字段，将该字段配置的路径改为上一步下载的量化数据集存放路径  

* Pulsar2 build

参考命令如下：

```
pulsar2 build --input SmolVLM-256M-Instruct_vision.onnx --config config.json --output_dir build-output --output_name SmolVLM-256M-Instruct_vision.axmodel --target_hardware AX650 --compiler.check 0
```
编译完成后将文件`build-output/SmolVLM-256M-Instruct_vision.axmodel` 放到 `../SmolVLM-256M-Instruct-AX650/`

## 二、转换 Language Model  

### 1. 转换Language Model  
执行命令
```
pulsar2 llm_build --input_path ../SmolVLM-256M-Instruct/ --output_path ../SmolVLM-256M-Instruct-AX650/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 320 --parallel 32 --chip AX650
```
其中 `prefill_len` 的长度就是 `prefill`阶段的最大token数，请根据实际情况设置这个值。
`kv_cache_len` `prefill`加上`decode`阶段token的总长度。

### 2. 从 Language model 中提取 token embeddings  

执行命令  
```
bash tools/embed_process.sh ../SmolVLM-256M-Instruct/ ../SmolVLM-256M-Instruct-AX650/
```
### 3. 拷贝配置文件
```
cp ../SmolVLM-256M-Instruct/*.json ../SmolVLM-256M-Instruct-AX650/
```

至此，整个模型转换完毕。将 `../SmolVLM-256M-Instruct-AX650/` 上传到爱芯的设备上准备运行。    