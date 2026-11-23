#!/usr/bin/env python3
"""Build and verify a product SRS1 v1 container directly from GGUF shards."""
import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import struct

ALIGN = 4096
SHARD_TABLE = 2048
LAYERS, EXPERTS, TOPK = 48, 512, 10
ROLES = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')
MODEL_NAME = 'Qwen3.8 Flash Next'
PREFIX = struct.Struct('<4sIQIIIIIIQQQQ')
COPY_BYTES = 16 * 1024 * 1024
SPACE_RESERVE = 256 * 1024 * 1024
_type_library = None


def type_info(kind):
    global _type_library
    if _type_library is None:
        root = Path(__file__).resolve().parent.parent
        _type_library = ctypes.CDLL(str(root / 'build-cuda/ggml/src/libggml-base.so'))
        _type_library.ggml_type_size.argtypes = [ctypes.c_int]
        _type_library.ggml_type_size.restype = ctypes.c_size_t
        _type_library.ggml_blck_size.argtypes = [ctypes.c_int]
        _type_library.ggml_blck_size.restype = ctypes.c_int64
    return _type_library.ggml_type_size(kind), _type_library.ggml_blck_size(kind)


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def parse_gguf(path):
    """Read metadata and tensor geometry without touching tensor payloads."""
    with open(path, 'rb') as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        magic, version, tensor_count, metadata_count = struct.unpack_from('<IIQQ', data)
        if magic != 0x46554747 or version != 3:
            raise ValueError(f'expected GGUF v3: {path}')
        cursor = 24
        def string():
            nonlocal cursor
            length = struct.unpack_from('<Q', data, cursor)[0]
            cursor += 8
            value = data[cursor:cursor + length].decode('utf-8')
            cursor += length
            return value
        formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        def value(kind, keep):
            nonlocal cursor
            if kind == 8:
                return string()
            if kind == 9:
                subtype, count = struct.unpack_from('<IQ', data, cursor)
                cursor += 12
                if subtype in formats and not keep:
                    cursor += count * struct.calcsize('<' + formats[subtype])
                    return None
                values = [value(subtype, keep) for _ in range(count)]
                return values if keep else None
            fmt = '<' + formats[kind]
            result = struct.unpack_from(fmt, data, cursor)[0] if keep else None
            cursor += struct.calcsize(fmt)
            return result
        metadata, offsets, entries = {}, {}, []
        for _ in range(metadata_count):
            key = string()
            kind = struct.unpack_from('<I', data, cursor)[0]
            cursor += 4
            if key == 'qwen4exp.attention.compress_ratios':
                subtype, count = struct.unpack_from('<IQ', data, cursor)
                if subtype not in (4, 5, 10, 11) or count != LAYERS:
                    raise ValueError('invalid compress_ratios array')
                offsets[key] = (cursor + 12, formats[subtype])
            value_start = cursor
            keep = key.startswith('qwen4exp.') or key in ('general.alignment', 'general.name', 'general.architecture', 'tokenizer.ggml.tokens')
            result = value(kind, keep)
            entries.append((key, kind, bytes(data[value_start:cursor])))
            if keep:
                metadata[key] = result
        alignment = int(metadata.get('general.alignment', 32))
        directory_start = cursor
        tensors = []
        for _ in range(tensor_count):
            name = string()
            dimensions = struct.unpack_from('<I', data, cursor)[0]
            cursor += 4
            if not 1 <= dimensions <= 4:
                raise ValueError(f'invalid tensor rank: {name}')
            shape = list(struct.unpack_from('<' + 'Q' * dimensions, data, cursor))
            cursor += 8 * dimensions
            kind, offset = struct.unpack_from('<IQ', data, cursor)
            cursor += 12
            tensors.append((name, shape + [1] * (4 - dimensions), kind, offset))
        return dict(path=path, size=os.path.getsize(path), data_off=align_up(cursor, alignment),
                    tensors=tensors, align=alignment, metadata=metadata, metadata_offsets=offsets, entries=entries, directory=bytes(data[directory_start:cursor]))


def validate_qsa(shards, tensors):
    """Authorize metadata normalization only for the matching QSA structure."""
    metadata = {}
    for shard in shards:
        metadata.update(shard['metadata'])
    def number(name):
        result = metadata.get('qwen4exp.' + name)
        if not isinstance(result, int) or result <= 0:
            raise ValueError(f'missing QSA metadata: {name}')
        return result
    width = number('embedding_length')
    dimension = number('attention.indexer.key_length')
    index_heads = number('attention.indexer.head_count')
    number('attention.indexer.top_k')
    heads = number('attention.head_count')
    kv_heads = number('attention.head_count_kv')
    head_dim = number('attention.key_length')
    expected_layers = list(range(3, LAYERS, 4))
    found = sorted({int(name.split('.')[1]) for name in tensors if '.indexer.' in name})
    if found != expected_layers:
        raise ValueError(f'indexer layers differ from QSA schedule: {found}')
    def shape(layer, suffix, expected):
        entry = tensors.get(f'blk.{layer}.{suffix}')
        if entry is None or entry[1] != expected + [1] * (4 - len(expected)):
            raise ValueError(f'invalid QSA tensor: layer={layer}, tensor={suffix}')
    for layer in expected_layers:
        shape(layer, 'indexer.k_proj.weight', [width, dimension])
        shape(layer, 'indexer.q_proj.weight', [width, dimension * index_heads])
        shape(layer, 'indexer.k_norm.weight', [dimension])
        shape(layer, 'indexer.q_norm.weight', [dimension])
        shape(layer, 'attn_output.weight', [head_dim * heads, width])
        shape(layer, 'attn_q_norm.weight', [head_dim])
        shape(layer, 'attn_k_norm.weight', [head_dim])
        if f'blk.{layer}.attn_q.weight' in tensors:
            shape(layer, 'attn_q.weight', [width, 2 * heads * head_dim])
            shape(layer, 'attn_k.weight', [width, kv_heads * head_dim])
            shape(layer, 'attn_v.weight', [width, kv_heads * head_dim])
        else:
            shape(layer, 'attn_qkv.weight', [width, 2 * heads * head_dim + 2 * kv_heads * head_dim])
    after = [4 if layer % 4 == 3 else 0 for layer in range(LAYERS)]
    corrections = []
    for owner, shard in enumerate(shards):
        key = 'qwen4exp.attention.compress_ratios'
        if key not in shard['metadata']:
            continue
        before = shard['metadata'][key]
        offset, fmt = shard['metadata_offsets'][key]
        corrections.append(dict(source=owner, key=key, offset=offset, before=before, after=after,
                                data=struct.pack('<48' + fmt, *after)))
    if not corrections:
        raise ValueError('missing compress_ratios metadata')
    return corrections


def rebuild_metadata(shard, corrections):
    """Serialize a complete GGUF directory; tensor offsets stay data-relative."""
    def string(value):
        data = value.encode('utf-8')
        return struct.pack('<Q', len(data)) + data
    entries = []
    found_name = False
    by_key = {item['key']: item for item in corrections}
    for key, kind, encoded in shard['entries']:
        if key == 'general.name':
            kind, encoded, found_name = 8, string(MODEL_NAME), True
        elif key in by_key:
            # Preserve the array element type/count, replace its validated values.
            encoded = encoded[:12] + by_key[key]['data']
        entries.append(string(key) + struct.pack('<I', kind) + encoded)
    if not found_name:
        entries.append(string('general.name') + struct.pack('<I', 8) + string(MODEL_NAME))
    header = struct.pack('<IIQQ', 0x46554747, 3, len(shard['tensors']), len(entries))
    result = header + b''.join(entries) + shard['directory']
    return result + bytes(align_up(len(result), shard['align']) - len(result))


def geometry(shape, kind):
    size, block = type_info(kind)
    if block <= 0 or shape[0] % block:
        raise ValueError('invalid GGML row geometry')
    return shape[0] // block * size * math.prod(shape[1:])


def plan(files):
    if not 1 <= len(files) <= 8:
        raise ValueError('expected 1..8 GGUF shards')
    enum_text = (Path(__file__).resolve().parent.parent / 'ggml/include/ggml.h').read_text()
    ids = {name: int(number) for name, number in re.findall(r'GGML_TYPE_(\w+)\s*=\s*(\d+)', enum_text)}
    allowed = {ids[name] for name in ('Q2_0', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS', 'IQ3_S',
                                      'IQ4_NL', 'IQ4_XS', 'Q8_0', 'Q4_0', 'Q4_K', 'Q5_K', 'Q5_1', 'Q6_K')}
    shards = [parse_gguf(Path(path)) for path in files]
    tensors = {}
    for index, shard in enumerate(shards):
        for name, shape, kind, offset in shard['tensors']:
            if name in tensors or not 0 <= kind < ids['COUNT']:
                raise ValueError(f'invalid tensor directory: {name}')
            length = geometry(shape, kind)
            if offset + length > shard['size'] - shard['data_off']:
                raise ValueError(f'tensor outside GGUF: {name}')
            tensors[name] = (index, shape, kind, offset, length)
    corrections = validate_qsa(shards, tensors)
    metadata = {key: value for shard in shards for key, value in shard['metadata'].items()}
    for key, expected in [('qwen4exp.block_count', 48), ('qwen4exp.expert_count', 512),
                          ('qwen4exp.expert_used_count', 10), ('qwen4exp.embedding_length', 2560)]:
        if metadata.get(key) != expected:
            raise ValueError(f'invalid Qwen3.8 structure: {key}')
    vocab = metadata.get('tokenizer.ggml.tokens', [])
    for index, token in enumerate(('<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>'), 248053):
        if len(vocab) <= index or vocab[index] != token:
            raise ValueError(f'invalid Qwen3.8 vision vocabulary: {index}')
    for owner, shard in enumerate(shards):
        shard['prefix'] = rebuild_metadata(shard, [item for item in corrections if item['source'] == owner])
        shard['data_delta'] = len(shard['prefix']) - shard['data_off']
        shard['output_size'] = shard['size'] + shard['data_delta']
        corrections.append(dict(source=owner, key='general.name', before=shard['metadata'].get('general.name'), after=MODEL_NAME))
    types, actual = [], []
    for layer in range(LAYERS):
        entries = [tensors[f'blk.{layer}.{role}'] for role in ROLES]
        for index, shape, kind, offset, length in entries:
            if kind not in allowed or shape[2] != EXPERTS or shape[3] != 1:
                raise ValueError(f'unsupported expert quant configuration: layer {layer}, type {kind}')
            types.append(kind)
            actual.append(length // EXPERTS)
        if entries[0][1] != entries[1][1] or entries[0][2] != entries[1][2]:
            raise ValueError(f'gate/up type and row layout differ: layer {layer}')
        gate_shape, down_shape = entries[0][1], entries[2][1]
        if gate_shape[:2] != list(reversed(down_shape[:2])):
            raise ValueError(f'gate/down shapes differ: layer {layer}')
    capacities = [max(actual[role::3]) for role in range(3)]
    offsets = [0, capacities[0], capacities[0] + capacities[1]]
    if any(offset % 256 for offset in offsets):
        raise ValueError('role offset is not 256-byte aligned')
    record = align_up(offsets[2] + capacities[2], ALIGN)
    position, dense_bytes = ALIGN, ALIGN
    for index, shard in enumerate(shards):
        shard['position'] = align_up(position, ALIGN)
        position = shard['position'] + shard['output_size']
        omitted = [(shard['data_off'] + offset, shard['data_off'] + offset + length)
                   for name, (owner, shape, kind, offset, length) in tensors.items()
                   if owner == index and name.endswith(ROLES)]
        cursor, ranges = 0, []
        for first, end in sorted(omitted) + [(shard['size'], shard['size'])]:
            if first < cursor:
                raise ValueError('overlapping expert tensors')
            if first > cursor:
                ranges.append((cursor, first - cursor))
            cursor = end
        shard['ranges'] = ranges
        shard['roles'] = sorted([(shard['data_off'] + offset, name, length) for name, (owner, shape, kind, offset, length) in tensors.items() if owner == index and name.endswith(ROLES)])
        dense_bytes += len(shard['prefix']) + sum(align_up(length + start % ALIGN, ALIGN) for start, length in ranges if start >= shard['data_off'])
        dense_bytes += sum(max(0, start + length - shard['data_off']) for start, length in ranges if start < shard['data_off'])
    live, peak, pending_bytes = {}, 0, 0
    for shard in shards:
        for offset, name, length in shard['roles']:
            layer = int(name.split('.')[1])
            live.setdefault(layer, []).append(length)
            pending_bytes += length
            peak = max(peak, pending_bytes)
            if len(live[layer]) == 3:
                peak = max(peak, pending_bytes + record * EXPERTS)
                pending_bytes -= sum(live.pop(layer))
    if peak:
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if peak + COPY_BYTES * 2 > available:
            raise RuntimeError(f'insufficient RAM for sequential pack: need={peak + COPY_BYTES * 2}, available={available}')
    expert_offset = align_up(position, ALIGN)
    header = bytearray(ALIGN)
    PREFIX.pack_into(header, 0, b'SRS1', 1, ALIGN, len(shards), LAYERS, EXPERTS, TOPK, 3, 0,
                     LAYERS * EXPERTS, expert_offset, record * LAYERS * EXPERTS, SHARD_TABLE)
    struct.pack_into('<6Q', header, PREFIX.size, *offsets, *capacities)
    struct.pack_into('<144I', header, PREFIX.size + 48, *types)
    for index, shard in enumerate(shards):
        struct.pack_into('<QQ', header, SHARD_TABLE + index * 16, shard['position'], shard['output_size'])
    return dict(shards=shards, tensors=tensors, types=types, actual=actual, role_offset=offsets,
                role_bytes=capacities, record_bytes=record, expert_offset=expert_offset, header=header,
                estimated_allocated_bytes=dense_bytes + record * LAYERS * EXPERTS, peak_buffer_bytes=peak, corrections=corrections)


def check_space(directory, remaining):
    free = shutil.disk_usage(directory).free
    if free < remaining + SPACE_RESERVE:
        raise RuntimeError(f'insufficient disk space: free={free}, remaining={remaining}, reserve={SPACE_RESERVE}')


def read_exact(fd, length, offset):
    data = os.pread(fd, length, offset)
    if len(data) != length:
        raise RuntimeError(f'short read at {offset}: {len(data)} of {length}')
    return data


def write_exact(fd, data, offset):
    if os.pwrite(fd, data, offset) != len(data):
        raise RuntimeError(f'short write at {offset}')


def verify_records(file, layout, expected):
    """Read back each physical record and hash roles in original tensor order."""
    record, offsets = layout['record_bytes'], layout['role_offset']
    for layer in range(LAYERS):
        hashes = [hashlib.sha256() for _ in ROLES]
        packet = read_exact(file, record * EXPERTS, layout['expert_offset'] + layer * EXPERTS * record)
        view = memoryview(packet)
        for expert in range(EXPERTS):
            for role in range(3):
                start = expert * record + offsets[role]
                hashes[role].update(view[start:start + layout['actual'][layer * 3 + role]])
        for role in range(3):
            if hashes[role].hexdigest() != expected[layer][role]:
                raise RuntimeError(f'role checksum mismatch: layer={layer}, role={role}')
        print(f'verified layer {layer + 1}/{LAYERS}', flush=True)


def pack(files, output):
    layout = plan(files)
    output = Path(output).resolve()
    if output.exists() or output in [Path(path).resolve() for path in files]:
        raise ValueError('output must be a new file; source GGUF is never modified')
    partial = output.with_name(output.name + '.partial')
    manifest_path = output.with_name(output.name + '.checksums.json')
    manifest_partial = output.with_name(output.name + '.checksums.json.partial')
    if partial.exists():
        raise ValueError(f'remove incomplete output before restarting: {partial}')
    remaining = layout['estimated_allocated_bytes']
    check_space(output.parent, remaining)
    manifest = dict(format='SRS1', version=1, checksum='sha256', record_bytes=layout['record_bytes'],
                    role_offset=layout['role_offset'], role_bytes=layout['role_bytes'], quant=layout['types'],
                    sources=[dict(path=str(s['path']), bytes=s['size']) for s in layout['shards']],
                    header_sha256=hashlib.sha256(layout['header']).hexdigest(),
                    expert_offset=layout['expert_offset'], expert_size=layout['record_bytes'] * LAYERS * EXPERTS,
                    shards=[dict(offset=shard['position'], bytes=shard['output_size'], data_delta=shard['data_delta']) for shard in layout['shards']],
                    dense=[], roles=[])
    published = False
    try:
        with ExitStack() as stack:
            sources = [stack.enter_context(open(shard['path'], 'rb', buffering=0)).fileno() for shard in layout['shards']]
            target = stack.enter_context(open(partial, 'x+b', buffering=0))
            fd = target.fileno()
            os.ftruncate(fd, layout['expert_offset'] + layout['record_bytes'] * LAYERS * EXPERTS)
            write_exact(fd, layout['header'], 0)
            remaining -= ALIGN
            pending = {}
            record = layout['record_bytes']
            role_hashes = [None] * LAYERS
            for owner, shard in enumerate(layout['shards']):
                source_hash = hashlib.sha256()
                prefix = shard['prefix']
                write_exact(fd, prefix, shard['position'])
                manifest['dense'].append(dict(offset=shard['position'], bytes=len(prefix), sha256=hashlib.sha256(prefix).hexdigest()))
                remaining = max(0, remaining - len(prefix))
                cursor = 0
                def dense(first, stop):
                    digest = hashlib.sha256()
                    for offset in range(first, stop, COPY_BYTES):
                        nonlocal remaining
                        check_space(output.parent, remaining)
                        original = read_exact(sources[owner], min(COPY_BYTES, stop - offset), offset)
                        source_hash.update(original)
                        start = max(offset, shard['data_off'])
                        data = original[start - offset:] if start < offset + len(original) else b''
                        if data:
                            digest.update(data)
                            write_exact(fd, data, shard['position'] + start + shard['data_delta'])
                            remaining = max(0, remaining - len(data))
                        copied = offset - first + len(data)
                        if copied // (1024 * 1024 * 1024) != (offset - first) // (1024 * 1024 * 1024):
                            print(f'dense source {owner + 1}: {copied}/{stop - first} bytes', flush=True)
                    first = max(first, shard['data_off'])
                    if stop > first:
                        manifest['dense'].append(dict(offset=shard['position'] + first + shard['data_delta'], bytes=stop - first, sha256=digest.hexdigest()))
                for offset, name, length in shard['roles']:
                    dense(cursor, offset)
                    data = read_exact(sources[owner], length, offset)
                    source_hash.update(data)
                    layer = int(name.split('.')[1])
                    role = ROLES.index(name.split('.', 2)[2])
                    pending.setdefault(layer, {})[role] = data
                    del data
                    cursor = offset + length
                    if len(pending[layer]) != 3:
                        continue
                    check_space(output.parent, remaining)
                    payloads = [memoryview(pending[layer][role]) for role in range(3)]
                    role_hashes[layer] = [hashlib.sha256(data).hexdigest() for data in payloads]
                    packet = bytearray(record * EXPERTS)
                    view = memoryview(packet)
                    for expert in range(EXPERTS):
                        for role in range(3):
                            size = layout['actual'][layer * 3 + role]
                            start = expert * record + layout['role_offset'][role]
                            view[start:start + size] = payloads[role][expert * size:(expert + 1) * size]
                    write_exact(fd, packet, layout['expert_offset'] + layer * EXPERTS * record)
                    remaining = max(0, remaining - len(packet))
                    del view, packet, payloads
                    del pending[layer]
                    print(f'packed layer {layer + 1}/{LAYERS}', flush=True)
                dense(cursor, shard['size'])
                manifest['sources'][owner]['sha256'] = source_hash.hexdigest()
                print(f'source {owner + 1}/{len(sources)} hashed and copied', flush=True)
            if pending or any(item is None for item in role_hashes):
                raise ValueError('incomplete expert payloads')
            manifest['roles'] = role_hashes
            manifest['metadata_corrections'] = [
                {key: value for key, value in change.items() if key != 'data'}
                for change in layout['corrections']]
            os.fsync(fd)
            for dense in manifest['dense']:
                digest = hashlib.sha256()
                for start in range(dense['offset'], dense['offset'] + dense['bytes'], COPY_BYTES):
                    digest.update(read_exact(fd, min(COPY_BYTES, dense['offset'] + dense['bytes'] - start), start))
                if digest.hexdigest() != dense['sha256']:
                    raise RuntimeError(f'dense checksum mismatch at {dense["offset"]}')
            verify_records(fd, layout, manifest['roles'])
            if read_exact(fd, ALIGN, 0) != layout['header']:
                raise RuntimeError('header readback mismatch')
        with open(manifest_partial, 'w') as file:
            json.dump(manifest, file, indent=2)
            file.write('\n'); file.flush(); os.fsync(file.fileno())
        os.link(partial, output)
        published = True
        os.replace(manifest_partial, manifest_path)
        partial.unlink()
    except BaseException:
        if published:
            output.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise
    print(f'verified SRS1 v1: {output}, record_bytes={layout["record_bytes"]}', flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', required=True)
    cli.add_argument('--plan', action='store_true', help='inspect GGUF geometry and required disk space without writing')
    cli.add_argument('shards', nargs='+')
    args = cli.parse_args()
    if args.plan:
        layout = plan(args.shards)
        print(json.dumps({key: layout[key] for key in ('record_bytes', 'role_offset', 'role_bytes', 'expert_offset', 'estimated_allocated_bytes')}, indent=2))
    else:
        pack(args.shards, args.output)
