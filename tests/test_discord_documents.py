"""commands/discord_documents.py: card documents' pictures (a Paint.NET file's own composite)."""
import base64
from io import BytesIO

from PIL import Image

import commands.discord_documents as dd


def test_a_paint_net_file_gives_its_composite():
    buf = BytesIO()
    Image.new("RGB", (12, 24), (200, 10, 10)).save(buf, "PNG")
    head = f'<pdnImage width="12" height="24"><custom><thumb png="{base64.b64encode(buf.getvalue()).decode()}" />' \
           '</custom></pdnImage>'.encode()
    data = b"PDN3" + len(head).to_bytes(3, "little") + head + b"\x00\x01rest"
    img = dd.pdn_thumb(data)
    assert img.size == (12, 24) and img.getpixel((3, 3))[:3] == (200, 10, 10)
    assert dd.pdn_thumb(b"PK\x03\x04") is None
