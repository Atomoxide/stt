# 语音识别转文字工具（API服务）

本项目为jianchang512/stt项目仅保留API的版本，并对部分依赖版本兼容性问题进行了修改。原项目地址：

https://github.com/jianchang512/stt


# 源码部署(Linux/Windows)

0. 要求 python 3.9->3.11

1. 使用git拉取源码到当前目录 ` git clone git@github.com:atomoxide/stt.git . `

2. 创建虚拟环境 `python -m venv .venv`，注意虚拟环境名称必须为`.venv`，否则无法使用项目的bash或powershell脚本启动

3. 激活环境，win下命令 `.\.venv\scripts\activate`，linux和Mac下命令 `source ./.venv/bin/activate`

4. 安装依赖: `pip install -r requirements.txt`

5. [下载模型压缩包](https://github.com/jianchang512/stt/releases/tag/0.0)，根据需要下载模型，下载后将压缩包里的文件夹放到项目根目录的 models 文件夹内

7. 执行  `.\run.ps1 ` (Windows) 或 `source ./run.sh` (Linux)。


# Api接口

接口地址: http://127.0.0.1:9977/api

请求方法: POST

请求参数:

    language: 语言代码:可选如下

    >
    > 中文：zh
    > 英语：en
    > 法语：fr
    > 德语：de
    > 日语：ja
    > 韩语：ko
    > 俄语：ru
    > 西班牙语：es
    > 泰国语：th
    > 意大利语：it
    > 葡萄牙语：pt
    > 越南语：vi
    > 阿拉伯语：ar
    > 土耳其语：tr
    >

    model: 模型名称，可选如下
    >
    > base 对应于 models/models--Systran--faster-whisper-base
    > small 对应于 models/models--Systran--faster-whisper-small
    > medium 对应于 models/models--Systran--faster-whisper-medium
    > large-v3 对应于 models/models--Systran--faster-whisper-large-v3
    >

    file: 音视频文件，二进制上传

Api 请求示例 (testApi.py)

```python
    import requests
    # 请求地址
    url = "http://127.0.0.1:9977/api"
    # 请求参数  file:音视频文件，language：语言代码，model：模型，response_format:text|json|srt
    # 返回 code==0 成功，其他失败，msg==成功为ok，其他失败原因，data=识别后返回文字
    files = {"file": open("C:/Users/c1/Videos/2.wav", "rb")}
    data={"language":"zh","model":"base"}
    response = requests.request("POST", url, timeout=600, data=data,files=files)
    print(response.json())
```


---

# CUDA 加速支持 （默认开启）

**安装CUDA工具** [详细安装方法](https://juejin.cn/post/7318704408727519270)

如果你的电脑拥有 Nvidia 显卡，先升级显卡驱动到最新，然后去安装对应的 
   [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)  和  [cudnn for CUDA11.X](https://developer.nvidia.com/rdp/cudnn-archive)。
   
   安装完成成，按`Win + R`,输入 `cmd`然后回车，在弹出的窗口中输入`nvcc --version`,确认有版本信息显示，类似该图
   ![image](https://github.com/jianchang512/pyvideotrans/assets/3378335/e68de07f-4bb1-4fc9-bccd-8f841825915a)

   然后继续输入`nvidia-smi`,确认有输出信息，并且能看到cuda版本号，类似该图
   ![image](https://github.com/jianchang512/pyvideotrans/assets/3378335/71f1d7d3-07f9-4579-b310-39284734006b)

    然后执行 `python testcuda.py`，如果提示成功，说明安装正确，否则请仔细检查重新安装
    
    默认使用 cpu 运算，如果确定使用英伟达显卡，并且配置好了cuda环境，请修改 set.ini 中 `devtype=cpu`为 `devtype=cuda`,并重新启动，可使用cuda加速

# 注意事项

0. 如果没有英伟达显卡或未配置好CUDA环境，不要使用 large/large-v3 模型，可能导致内存耗尽死机
1. 中文在某些情况下会输出繁体字
2. 有时会遇到“cublasxx.dll不存在”的错误，此时需要下载 cuBLAS，然后将dll文件复制到系统目录下，[点击下载 cuBLAS](https://github.com/jianchang512/stt/releases/download/0.0/cuBLAS_win.7z)，解压后将里面的dll文件复制到 C:/Windows/System32下
3. 如果控制台出现"[W:onnxruntime:Default, onnxruntime_pybind_state.cc:1983 onnxruntime::python::CreateInferencePybindStateModule] Init provider bridge failed.", 可忽略，不影响使用
4. 默认使用 cpu 运算，如果确定使用英伟达显卡，并且配置好了cuda环境，请修改 set.ini 中 `devtype=cpu`为 `devtype=cuda`,并重新启动，可使用cuda加速



5. 尚未执行完毕就闪退

如果启用了cuda并且电脑已安装好了cuda环境，但没有手动安装配置过cudnn，那么会出现该问题，去安装和cuda匹配的cudnn。比如你安装了cuda12.3，那么就需要下载cudnn for cuda12.x压缩包，然后解压后里面的3个文件夹复制到cuda安装目录下。具体教程参考 https://juejin.cn/post/7318704408727519270

如果cudnn按照教程安装好了仍闪退，那么极大概率是GPU显存不足，可以改为使用 medium模型，显存不足8G时，尽量避免使用largev-3模型，尤其是视频大于20M时，否则可能显存不足而崩溃

# 相关联项目

[视频翻译配音工具:翻译字幕并配音](https://github.com/jianchang512/pyvideotrans)

[声音克隆工具:用任意音色合成语音](https://github.com/jianchang512/clone-voice)

[人声背景乐分离:极简的人声和背景音乐分离工具，本地化网页操作](https://github.com/jianchang512/vocal-separate)

# 致谢

本项目基于jianchang512/stt项目，原项目地址：

https://github.com/jianchang512/stt

原项目主要依赖的其他项目

1. https://github.com/SYSTRAN/faster-whisper
2. https://github.com/pallets/flask
3. https://ffmpeg.org/
4. https://layui.dev

