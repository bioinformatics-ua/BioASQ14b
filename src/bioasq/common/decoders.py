"""Decoders for structured domain objects.

Provides pre-configured msgspec decoders for parsing BioASQ JSON data
into strictly typed domain structures.
"""

from __future__ import annotations

import msgspec

from bioasq.common.types import Document

document_decoder = msgspec.json.Decoder(Document)
