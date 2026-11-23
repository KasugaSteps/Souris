---
library_name: souris
language:
  - zh
tags:
  - qwen
  - souris
  - iq2_xxs
  - mtp
  - cuda
  - single-file
---
<p align="center">
  <img src="assets/souris-logo.png" alt="Souris" width="340">
</p>

<h1 align="center">Souris · Qwen3.8 Flash Next</h1>

<p align="center">
  在 24 GB 显卡上运行 32K 上下文的 Qwen3.8 Flash Next。<br>
  IQ2_XXS / Q4_0 混合量化，全 GPU 推理。
</p>

<p align="center">
  <a href="#性能">性能</a> ·
  <a href="#全-gpu">全 GPU</a> ·
  <a href="#安装">安装</a> ·
  <a href="#权重">权重</a> ·
  <a href="#限制">限制</a>
</p>

---

## 这是什么

Souris 是 Qwen3.8 Flash Next 的本地推理运行引擎。

它在一张 24 GB 显卡上运行 33,280 token 上下文，支持 MTP，对外提供 OpenAI 兼容接口。

## 性能

RTX 4090 24 GB 实测：

| 阶段    | 吞吐（tok/s） |
| ------- | ------------: |
| Prefill |        1855.4 |
| Decode  |          58.0 |

## 全 GPU

一张 24 GB 显卡放不下 118 GB 模型。常见方案是把一部分 expert 交给 CPU 计算；Souris 没有把计算拆到 CPU，而是把 expert 放在 SSD，按需读入 GPU，Prefill 和 Decode 全部在 GPU 上完成。

### Prefill 不重复读盘

长 prompt 按 chunk 切块。逐块取 expert，会让同一层反复读同一个 expert。Souris 先合并同一层所有 chunk 的路由，得到这一层真正需要的 expert 集合；每个 expert 只读一次，按磁盘顺序读，16 个一组送入 GPU 做 MMQ。读盘从“每块一次”降到“每层一次”。

### Decode 不被 Prefill 冲掉

Decode 的热 expert 很少，但会反复使用。Souris 把显存里的 expert 缓存分成 protected（保护）和 probation（试用）两段：命中的 expert 进入 protected，protected 满了就把最久未使用的降到 probation；Decode 正在使用的 expert 固定，MTP 验证的临时 expert 只使用空闲或 probation，不驱逐 protected。

结果：Prefill 只读必要的 expert，Decode 留下常用的 expert，24 GB 单卡可以全 GPU 跑完整模型。

## 安装

### 1. 下载权重

权重地址：

https://huggingface.co/linyaocai/Qwen_38_flash_for_Souris

```bash
hf download linyaocai/Qwen_38_flash_for_Souris \
  --local-dir ./weights
```

权重按 10 GB 分片存放，下载后合并：

```bash
cat weights/souris-iq2xxs.bin.parts/part-* \
  > weights/souris-iq2xxs.bin
```

视觉编码器权重位于：

```text
weights/vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
```

### 2. 编译

```bash
sudo apt install -y build-essential cmake libssl-dev
cd code
cmake -S . -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --target souris -j8
cd ..
```

### 3. 配置

```bash
cp code/souris.toml.example souris.toml
```

权重路径已经写在示例里；如果目录不同，改 `souris.toml` 即可。

### 4. 启动后端

```bash
./code/build-cuda/bin/souris serve --config souris.toml
```

### 5. 启动前端

```bash
python3 -m venv code/frontend/.venv
code/frontend/.venv/bin/pip install -r code/frontend/requirements.txt

code/frontend/.venv/bin/python code/frontend/server.py \
  --backend http://127.0.0.1:18080 \
  --port 18081
```

OpenAI 接口：

```text
http://127.0.0.1:18081/v1
```

### 6. 接入 Pi

```bash
pi --provider souris --model souris-iq2xxs \
  --session-id souris-local-01
```

```json
{
  "baseUrl": "http://127.0.0.1:18081/v1",
  "apiKey": "local",
  "model": "souris-iq2xxs"
}
```

## 权重

复现所需权重全部在 Hugging Face：

https://huggingface.co/linyaocai/Qwen_38_flash_for_Souris

```text
souris-iq2xxs.bin.parts/part-00 ... part-11   118.7 GB
souris-iq2xxs.bin（合并后）                   118.7 GB
mtp/weights.bin                                1.37 GiB
mtp/manifest.json                              MTP 布局和校验
vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf    616.7 MB
```

`souris-iq2xxs.bin` 是 SRS1 v1 容器，不是 GGUF，也不是 safetensors。

## 限制

- 只验证了 `(IQ2_XXS, IQ2_XXS, Q4_0)` 这一组目标量化。其他类型有代码路径，但没有完成性能验证。
- 短 prompt 下 MTP 可能变慢，不能当作普遍加速。

## 致谢

Qwen 团队提供基础模型。GGML 提供量化、MMQ 和底层算子。

---

<p align="center">
  <sub>SOURIS</sub>
</p>
