from __future__ import annotations

import msgspec

from bioasq.common.types import Document

document_decoder = msgspec.json.Decoder(Document)
