"""Embed a pipeline (.js) inside an .onnx model, so the model brings its own analysis when it is opened.

    python scripts/embed_pipeline.py MODEL.onnx PIPELINE.js [OUT.onnx]   embed (default OUT: MODEL_with_pipeline.onnx)
    python scripts/embed_pipeline.py --extract MODEL.onnx [OUT.js]        write the embedded pipeline back out
    python scripts/embed_pipeline.py --remove MODEL.onnx OUT.onnx          copy the model without its pipeline

The pipeline's source goes into the model's metadata under the key below, next to any other metadata (the
Ultralytics keys, an embedded detector.manifest), which are kept. The graph itself is not touched.
Re-run after every change to the pipeline. Requires the `onnx` package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx

KEY = "model-studio.pipeline"  # must match PIPELINE_METADATA_KEY in packages/core/src/onnxMeta.ts


def set_meta(model: onnx.ModelProto, key: str, value: str | None) -> None:
    props = [p for p in model.metadata_props if p.key != key]
    del model.metadata_props[:]
    model.metadata_props.extend(props)
    if value is not None:
        model.metadata_props.add(key=key, value=value)


def main(argv: list[str]) -> None:
    if argv[:1] == ["--extract"] and len(argv) in (2, 3):
        model = onnx.load(argv[1], load_external_data=False)
        source = next((p.value for p in model.metadata_props if p.key == KEY), None)
        if source is None:
            sys.exit(f"{argv[1]} has no embedded pipeline")
        out = Path(argv[2]) if len(argv) == 3 else Path(argv[1]).with_suffix(".pipeline.js")
        out.write_text(source, encoding="utf-8", newline="")  # exactly as embedded
        print(f"wrote {out} ({len(source)} chars)")
    elif argv[:1] == ["--remove"] and len(argv) == 3:
        model = onnx.load(argv[1])
        set_meta(model, KEY, None)
        onnx.save(model, argv[2])
        print(f"wrote {argv[2]} without a pipeline")
    elif len(argv) in (2, 3) and not argv[0].startswith("--"):
        model_path, pipeline_path = Path(argv[0]), Path(argv[1])
        out = Path(argv[2]) if len(argv) == 3 else model_path.with_name(model_path.stem + "_with_pipeline.onnx")
        source = pipeline_path.read_text(encoding="utf-8")
        if "export default" not in source:
            sys.exit(f"{pipeline_path} doesn't look like a pipeline (no 'export default')")
        model = onnx.load(model_path)
        set_meta(model, KEY, source)
        onnx.save(model, out)
        print(f"wrote {out}: {model_path.name} + {pipeline_path.name} ({len(source)} chars)")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
