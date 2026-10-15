import { builtinDecoderNames, getDecoder, type ModelConfig } from "@detect/core";
import { useState } from "react";

/** Decoder and pipeline option sliders (thresholds etc.) are hidden for now; set true to bring them back. */
export const SHOW_OPTIONS = false;

interface Props {
  config: ModelConfig;
  /** Name of the loaded pipeline when it supplies its own decoder; the config's decoder is then unused. */
  decoderFromPipeline?: string;
  onChange(config: ModelConfig): void;
}

/** Edits the decoder and its options. Options are generated from the decoder's defaults. */
export function ConfigEditor({ config, decoderFromPipeline, onChange }: Props) {
  const decoder = decoderFromPipeline ? null : safeDecoder(config.decoder);
  const options = decoderFromPipeline ? {} : { ...decoder?.defaults, ...config.options };
  const setOption = (key: string, value: unknown) => onChange({ ...config, options: { ...config.options, [key]: value } });
  const names = builtinDecoderNames();
  if (!names.includes(config.decoder)) names.push(config.decoder);

  return (
    <>
      {decoderFromPipeline ? (
        <p className="hint">
          Decoder: from pipeline <b>{decoderFromPipeline}</b>
        </p>
      ) : (
        <label className="field">
          <span>Decoder</span>
          <select value={config.decoder} onChange={(e) => onChange({ ...config, decoder: e.target.value, options: {} })}>
            {names.map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
      )}

      {SHOW_OPTIONS && Object.keys(options).length > 0 && <h3>Options</h3>}
      {SHOW_OPTIONS && Object.entries(options).map(([key, value]) => (
        <OptionField
          key={`${config.decoder}.${key}`}
          name={key}
          value={value}
          range={decoder?.ranges?.[key]}
          onChange={(v) => setOption(key, v)}
        />
      ))}
      <p className="hint">{config.labels?.length ? `${config.labels.length} labels` : "No labels (class ids shown)"}</p>
    </>
  );
}

function safeDecoder(name: string) {
  try {
    return getDecoder(name);
  } catch {
    return null;
  }
}

export function OptionField({ name, value, range, onChange }: { name: string; value: unknown; range?: [number, number]; onChange(v: unknown): void }) {
  if (typeof value === "boolean") {
    return (
      <label className="check">
        <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} /> {name}
      </label>
    );
  }
  if (typeof value === "number" && (range || /threshold/i.test(name))) {
    const [min, max] = range ?? [0, 1];
    return (
      <label className="field">
        <span>
          {name} <b>{value.toFixed(2)}</b>
        </span>
        <input type="range" min={min} max={max} step={(max - min) / 100} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      </label>
    );
  }
  if (typeof value === "number") return <NumberField label={name} value={value} step="any" onChange={onChange} />;
  if (typeof value === "string") {
    return (
      <label className="field">
        <span>{name}</span>
        <input value={value} onChange={(e) => onChange(e.target.value)} />
      </label>
    );
  }
  return <JsonField label={name} value={value} onChange={onChange} />;
}

function NumberField({ label, value, step, onChange }: { label: string; value: number; step?: string; onChange(v: number): void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        step={step ?? "1"}
        value={value}
        onChange={(e) => e.target.value !== "" && onChange(Number(e.target.value))}
      />
    </label>
  );
}

/** Free-form JSON value, applied on blur. */
function JsonField({ label, value, onChange }: { label: string; value: unknown; onChange(v: unknown): void }) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState("");
  return (
    <label className="field">
      <span>
        {label} {error && <em className="err">{error}</em>}
      </span>
      <input
        value={text ?? JSON.stringify(value)}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => {
          if (text === null) return;
          try {
            onChange(JSON.parse(text));
            setText(null);
            setError("");
          } catch (e) {
            setError((e as Error).message);
          }
        }}
      />
    </label>
  );
}
