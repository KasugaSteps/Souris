"""Static image input: bounded download, decode and fixed RGB8 letterbox."""
import asyncio
import base64
import binascii
import copy
import hashlib
import io
import ipaddress
import socket
from urllib.parse import urlsplit, urljoin

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

SIZE = 768
MAX_BYTES = 10_000_000
MAX_PIXELS = 40_000_000
VERSION = b"souris-letterbox-rgb8-v1"
PLACEHOLDER = "<|vision_start|>" + "<|image_pad|>" * 576 + "<|vision_end|>"


class ImageError(ValueError):
    pass


def validate_content(content):
    if content is None or isinstance(content, str):
        return
    if not isinstance(content, list):
        raise ImageError("invalid message content")
    for part in content:
        if not isinstance(part, dict):
            raise ImageError("invalid message content")
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str) and not set(part) - {"type", "text"}:
            continue
        if kind == "image_url":
            value = part.get("image_url")
            if set(part) != {"type", "image_url"} or not isinstance(value, dict) or not isinstance(value.get("url"), str) or set(value) - {"url", "detail"}:
                raise ImageError("invalid image data")
            continue
        if kind == "image":
            if set(part) - {"type", "data", "mimeType"} or not isinstance(part.get("data"), str) or not isinstance(part.get("mimeType"), str):
                raise ImageError("invalid image data")
            continue
        raise ImageError("unsupported image format" if kind != "text" else "invalid message content")


def _animated(data, kind):
    offset = 8 if kind == "PNG" else 12
    while offset + 8 <= len(data):
        if kind == "PNG":
            size, name = int.from_bytes(data[offset:offset+4], "big"), data[offset+4:offset+8]
            if name == b"acTL":
                return True
            if name == b"IEND":
                return False
            offset += size + 12
        else:
            name, size = data[offset:offset+4], int.from_bytes(data[offset+4:offset+8], "little")
            if name in (b"ANIM", b"ANMF") or (name == b"VP8X" and offset + 8 < len(data) and data[offset+8] & 2):
                return True
            offset += 8 + size + (size & 1)
    return False


def decode_image(data):
    if len(data) > MAX_BYTES:
        raise ImageError("image too large")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        kind = "PNG"
    elif data.startswith(b"\xff\xd8\xff"):
        kind = "JPEG"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        kind = "WEBP"
    elif data.startswith(b"BM"):
        kind = "BMP"
    else:
        raise ImageError("unsupported image format")
    if kind in ("PNG", "WEBP") and _animated(data, kind):
        raise ImageError("unsupported image format")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format != kind or getattr(source, "n_frames", 1) != 1 or getattr(source, "is_animated", False):
                raise ImageError("unsupported image format")
            width, height = source.size
            if width * height > MAX_PIXELS or max(width, height) > 8192 or min(width, height) < 16:
                raise ImageError("image too large" if width * height > MAX_PIXELS or max(width, height) > 8192 else "invalid image size")
            source.load()
            oriented = ImageOps.exif_transpose(source)
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
            rgb = Image.alpha_composite(background, rgba).convert("RGB")
            scale = SIZE / max(rgb.size)
            shape = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            rgb = rgb.resize(shape, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (SIZE, SIZE), (127, 127, 127))
            canvas.paste(rgb, ((SIZE - rgb.width) // 2, (SIZE - rgb.height) // 2))
            raw = canvas.tobytes()
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageError("invalid image data") from exc
    return {"format": "rgb8", "width": SIZE, "height": SIZE, "data": base64.b64encode(raw).decode()}, hashlib.sha256(raw + VERSION).hexdigest()


async def _download(url):
    # Resolve and connect to the validated address, while retaining TLS SNI and
    # HTTP Host. Redirects must repeat validation; DNS rebinding cannot change
    # the destination after the check. Environment proxies are not used.
    async with httpx.AsyncClient(timeout=10, trust_env=False, follow_redirects=False) as client:
        for _ in range(4):
            try:
                parts = urlsplit(url)
                if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
                    raise ImageError("invalid image URL")
                port = parts.port or (443 if parts.scheme == "https" else 80)
                addresses = await asyncio.get_running_loop().getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
                values = [ipaddress.ip_address(item[4][0]) for item in addresses]
                if not values or any(not value.is_global or value.is_multicast for value in values):
                    raise ImageError("private image URL is not allowed")
                original = httpx.URL(url)
                pinned = original.copy_with(host=str(values[0]))
                host = original.netloc.decode()
                async with client.stream("GET", pinned, headers={"Host": host}, extensions={"sni_hostname": original.host}) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        url = urljoin(url, response.headers.get("location", ""))
                        continue
                    if response.status_code != 200:
                        raise ImageError("image download failed")
                    length = response.headers.get("content-length")
                    if length and int(length) > MAX_BYTES:
                        raise ImageError("image too large")
                    data = bytearray()
                    async for block in response.aiter_bytes():
                        if len(data) + len(block) > MAX_BYTES:
                            raise ImageError("image too large")
                        data.extend(block)
                    return bytes(data)
            except ImageError:
                raise
            except (ValueError, httpx.HTTPError, OSError) as exc:
                raise ImageError("image download failed") from exc
    raise ImageError("too many image redirects")


async def load_image(part):
    if part["type"] == "image":
        url = "data:" + part["mimeType"] + ";base64," + part["data"]
    else:
        url = part["image_url"]["url"]
    if url.startswith("data:"):
        header, separator, encoded = url.partition(",")
        if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ImageError("invalid image data")
        if len(encoded) > (MAX_BYTES + 2) // 3 * 4:
            raise ImageError("image too large")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ImageError("invalid image data") from exc
    else:
        try:
            data = await asyncio.wait_for(_download(url), timeout=10)
        except asyncio.TimeoutError as exc:
            raise ImageError("image download timeout") from exc
    return await asyncio.to_thread(decode_image, data)


async def prepare_images(body, chat):
    if not chat:
        return body
    count = sum(1 for message in body["messages"] if isinstance(message.get("content"), list)
                for part in message["content"] if part["type"] in ("image_url", "image"))
    if count > 4:
        raise ImageError("too many images")
    if not count:
        return body
    prepared = copy.deepcopy(body)
    images = []
    for message in prepared["messages"]:
        if not isinstance(message.get("content"), list):
            continue
        pieces = []
        for part in message["content"]:
            if part["type"] == "text":
                pieces.append(part["text"])
            else:
                raw, _ = await load_image(part)
                images.append(raw)
                pieces.append(PLACEHOLDER)
        message["content"] = "".join(pieces)
    prepared["_images"] = images
    return prepared
