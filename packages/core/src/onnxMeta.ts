// Available in browsers, Node and Hermes; declared here because core builds without DOM types.
declare class TextDecoder {
  decode(input: Uint8Array): string;
}

/**
 * Reads ModelProto.metadata_props (field 14) from raw .onnx bytes without a protobuf library.
 * Only top-level fields are scanned; the graph is skipped by length.
 */
/** ONNX metadata key holding a pipeline's source, written by scripts/embed_pipeline.py. */
export const PIPELINE_METADATA_KEY = "model-studio.pipeline";

export function readOnnxMetadata(bytes: Uint8Array): Record<string, string> {
  const meta: Record<string, string> = {};
  const r = new Reader(bytes);
  while (r.pos < bytes.length) {
    const tag = r.varint();
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    if (field === 14 && wire === 2) {
      const entry = new Reader(r.bytes());
      let key = "";
      let value = "";
      while (entry.pos < entry.buf.length) {
        const t = entry.varint();
        if (t === 10) key = entry.string();
        else if (t === 18) value = entry.string();
        else entry.skip(t % 8);
      }
      meta[key] = value;
    } else {
      r.skip(wire);
    }
  }
  return meta;
}

class Reader {
  pos = 0;
  constructor(readonly buf: Uint8Array) {}

  varint(): number {
    let result = 0;
    let mul = 1;
    for (;;) {
      if (this.pos >= this.buf.length) throw new Error("Truncated ONNX file");
      const b = this.buf[this.pos++];
      result += (b & 0x7f) * mul;
      if (b < 0x80) return result;
      mul *= 128;
    }
  }

  bytes(): Uint8Array {
    const len = this.varint();
    const out = this.buf.subarray(this.pos, this.pos + len);
    this.pos += len;
    return out;
  }

  string(): string {
    return new TextDecoder().decode(this.bytes());
  }

  skip(wire: number) {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) {
      const len = this.varint(); // must advance pos before adding
      this.pos += len;
    }
    else if (wire === 5) this.pos += 4;
    else throw new Error(`Unsupported protobuf wire type ${wire}`);
  }
}
