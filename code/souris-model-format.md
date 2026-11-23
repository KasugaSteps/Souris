# souris-model.bin 格式

路径：

```text
/root/souris-model.bin
```

逻辑大小：

```text
165,126,254,592 B
```

物理占用：

```text
约 93 GiB
```

原因：

```text
原始 3 个 GGUF 分片完整复制
-> expert tensor 区间 punch hole
-> 追加 packed expert store
```

---

## 1. 文件布局

```text
0                          SRS1 header (4096 B)
4096                       shard 0
39662891008                shard 1
79262674944                shard 2
97681846272                expert store
165126254592               EOF
```

shard 段仍是原始 GGUF 字节，  
但 expert tensor 的数据区间已经是洞（读出来全 0）。

---

## 2. Header

小端。

```text
offset  size  field
0       4     magic = "SRS1"
4       4     version = 1
8       8     header_size = 4096
16      4     shard_count = 3
20      4     reserved
24      8     expert_count = 24576
32      4     layer_count = 48
36      4     experts_per_layer = 512
40      8     record_size = 2744320
48      8     expert_stride = 2741760
56      8     expert_offset = 97681846272
64      8     expert_size = 67444408320
72      8     shard_table_offset = 128
```

shard table，从 128 开始，每项 16 字节：

```text
u64 offset
u64 size
```

---

## 3. Shard 段

每个 shard 是原始 GGUF 文件原样。

读取方式：

```text
gguf bytes = file[shard_offset : shard_offset + shard_size]
```

GGUF 内部的 `data_offset` 是相对于 shard 起点。

expert tensor 仍然在 tensor directory 里，  
但对应数据区被 punch hole，读出来是 0。

runtime 必须：

```text
dense tensor -> 从 shard 段读
expert tensor -> 从 expert store 读
不要读 shard 里的 expert 数据
```

---

## 4. Expert store

record 索引：

```text
index = layer * 512 + expert
offset = expert_offset + index * record_size
```

record 内：

```text
offset 0        gate expert slice      870400 B
offset 900864   up expert slice        870400 B
offset 1801728  down expert slice      921600 B
offset 2723328  padding to 2744320
```

注意：

```text
expert_stride = 2741760   GPU slot stride，无 4K padding
record_size   = 2744320   store record stride，4K 对齐
```

H2D：

```text
cudaMemcpyAsync(vram_slot, record + role_offset, role_size)
```

或者：

```text
cudaMemcpyAsync(vram_slot, record, expert_stride)
```

后者会多拷 padding，但源和目标都连续。

---

## 5. 角色

```text
gate: ffn_gate_exps.weight  type 23 IQ4_XS
up:   ffn_up_exps.weight    type 23 IQ4_XS
down: ffn_down_exps.weight  type 20 IQ4_NL
```

每个 expert：

```text
gate 870400 B
up   870400 B
down 921600 B
```

---

## 6. 校验

已抽样比较：

```text
layer 0 expert 0
layer 0 expert 511
layer 24 expert 256
layer 47 expert 511
```

每个 role 与原始 GGUF 对应 expert slice 字节一致。

---

## 7. 原文件

转换完成后：

```text
/root/weights/Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf
/root/weights/Qwen3.8-Flash-Next-IQ4_XS-00002-of-00003.gguf
/root/weights/Qwen3.8-Flash-Next-IQ4_XS-00003-of-00003.gguf
```

将被删除。


## SRS1 v2 extension

The original 80-byte container prefix and segment table at byte 128 remain.
Container version is 2. At byte 256 the 80-byte little-endian Srs1Layout stores
magic SRS1, version 2, record_bytes u64, stride u64, role_count u32, reserved
u32, role_offset[3] u64 and role_bytes[3] u64. Role byte values are maximum
capacities across layers. Actual payload byte counts and quantization types
come from the embedded GGUF directories. Padding in each record is zero.
Segments may number 1 through 8. Expert records are 4096-byte aligned; GPU
stride and role offsets are 256-byte aligned. No expert tensor hole is read.
Use tools/srs1.py to pack original shards or upgrade the v1 header in place.
