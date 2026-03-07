"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Detection {
  bbox: number[];
  class_id: number;
  class_name: string;
  confidence: number;
  furniture_type: string | null;
  condition: string | null;
}

interface DetectionResult {
  detections: Detection[];
  count: number;
  inference_time_ms: number;
  image_width: number;
  image_height: number;
  annotated_image: string;
  model_classes?: string[];
}

interface HealthInfo {
  status: string;
  model_loaded: boolean;
  model_path: string | null;
  classes: string[];
}

const conditionStyles: Record<string, { bg: string; border: string; dot: string }> = {
  broken:  { bg: "bg-red-50 dark:bg-red-950/30", border: "border-red-500", dot: "bg-red-500" },
  wornout: { bg: "bg-amber-50 dark:bg-amber-950/30", border: "border-amber-500", dot: "bg-amber-500" },
  damaged: { bg: "bg-red-50 dark:bg-red-950/30", border: "border-red-400", dot: "bg-red-400" },
};
const defaultStyle = { bg: "bg-slate-50 dark:bg-slate-800/30", border: "border-accent", dot: "bg-accent" };

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.5);
  const [dragActive, setDragActive] = useState(false);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API_URL}/`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG, etc.)");
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  const runDetection = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch(
        `${API_URL}/detect/full?confidence=${confidence}`,
        { method: "POST", body: formData }
      );

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Server error: ${res.status}`);
      }

      const data: DetectionResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to connect to backend. Is the API server running?"
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const grouped = result?.detections.reduce(
    (acc, d) => {
      const key = d.class_name;
      if (!acc[key]) acc[key] = [];
      acc[key].push(d);
      return acc;
    },
    {} as Record<string, Detection[]>
  );

  const backendOnline = health?.status === "ok";

  return (
    <div className="min-h-screen bg-background">
      {/* ── Header ── */}
      <header className="sticky top-0 z-50 border-b border-border bg-card/80 backdrop-blur-lg">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-white font-bold">
              V
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">VisionAI</h1>
              <p className="text-[11px] leading-none text-muted">
                Furniture Condition Detection
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4 text-xs text-muted">
            <span className="flex items-center gap-1.5">
              <span
                className={`inline-block h-2 w-2 rounded-full ${backendOnline ? "bg-success" : "bg-danger"}`}
              />
              {backendOnline ? "API Online" : "API Offline"}
            </span>
            {health?.model_path && (
              <span className="hidden sm:inline truncate max-w-[200px]">
                {health.model_path.split(/[\\/]/).pop()}
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        <div className="grid gap-8 lg:grid-cols-3">
          {/* ── Left: Upload + Image ── */}
          <div className="space-y-5 lg:col-span-2">
            <div
              className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 ${
                dragActive
                  ? "border-accent bg-accent/5 scale-[1.01]"
                  : "border-border hover:border-accent/40"
              } ${!previewUrl ? "p-16" : "p-4"}`}
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
            >
              {!previewUrl ? (
                <div className="flex flex-col items-center gap-4 text-center">
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-accent/10">
                    <svg className="h-8 w-8 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-lg font-semibold">
                      Drop an image here, or{" "}
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="text-accent hover:text-accent-light underline underline-offset-2"
                      >
                        browse
                      </button>
                    </p>
                    <p className="mt-1 text-sm text-muted">
                      Supports JPEG, PNG, WebP
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative overflow-hidden rounded-xl bg-black/5 dark:bg-white/5">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={result?.annotated_image || previewUrl}
                      alt="Detection preview"
                      className="mx-auto max-h-[520px] w-full object-contain"
                    />
                    {loading && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                        <div className="flex flex-col items-center gap-3">
                          <div className="h-10 w-10 animate-spin rounded-full border-4 border-white/20 border-t-white" />
                          <span className="text-sm font-medium text-white">
                            Analyzing image...
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    <button
                      onClick={runDetection}
                      disabled={loading || !backendOnline}
                      className="flex-1 rounded-xl bg-accent px-6 py-3 text-sm font-semibold text-white transition-all hover:bg-accent-light disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? "Detecting..." : result ? "Re-run Detection" : "Run Detection"}
                    </button>
                    <button
                      onClick={reset}
                      className="rounded-xl border border-border px-6 py-3 text-sm font-medium transition-colors hover:bg-card"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleFile(e.target.files[0]);
                }}
              />
            </div>

            {error && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/30 dark:text-red-400">
                {error}
              </div>
            )}

            {!backendOnline && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-400">
                Backend API is offline. Start it with:{" "}
                <code className="rounded bg-amber-100 px-1.5 py-0.5 font-mono text-xs dark:bg-amber-900/50">
                  cd backend &amp;&amp; python main.py
                </code>
              </div>
            )}
          </div>

          {/* ── Right: Controls + Results ── */}
          <div className="space-y-5">
            {/* Confidence */}
            <div className="rounded-2xl border border-border bg-card p-5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
                Settings
              </h3>
              <div className="mt-4 space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Confidence Threshold</span>
                  <span className="font-mono font-semibold text-accent">
                    {(confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.05"
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                  className="w-full accent-accent"
                />
                <div className="flex justify-between text-[10px] text-muted">
                  <span>10%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>

            {/* Stats */}
            {result && (
              <div className="rounded-2xl border border-border bg-card p-5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
                  Summary
                </h3>
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <Stat value={String(result.count)} label="Objects Found" />
                  <Stat
                    value={`${result.inference_time_ms.toFixed(0)}`}
                    unit="ms"
                    label="Inference"
                  />
                  <Stat
                    value={`${result.image_width}x${result.image_height}`}
                    label="Image Size"
                  />
                  <Stat
                    value={String(Object.keys(grouped || {}).length)}
                    label="Unique Classes"
                  />
                </div>
              </div>
            )}

            {/* Detection list */}
            {result && grouped && (
              <div className="rounded-2xl border border-border bg-card p-5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
                  Detections
                </h3>
                {result.count === 0 ? (
                  <p className="mt-4 text-sm text-muted">
                    No objects detected. Try lowering the confidence threshold.
                  </p>
                ) : (
                  <div className="mt-4 max-h-[400px] space-y-4 overflow-y-auto pr-1">
                    {Object.entries(grouped).map(([cls, dets]) => (
                      <div key={cls}>
                        <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-muted">
                          {cls} ({dets.length})
                        </p>
                        <div className="space-y-1.5">
                          {dets.map((d, i) => {
                            const style =
                              (d.condition && conditionStyles[d.condition]) ||
                              defaultStyle;
                            return (
                              <div
                                key={i}
                                className={`flex items-center justify-between rounded-lg border-l-4 ${style.border} ${style.bg} px-3 py-2`}
                              >
                                <div className="flex items-center gap-2">
                                  <span
                                    className={`inline-block h-2 w-2 rounded-full ${style.dot}`}
                                  />
                                  <span className="text-sm font-medium capitalize">
                                    {d.condition || d.class_name}
                                  </span>
                                </div>
                                <span className="font-mono text-xs font-semibold">
                                  {(d.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Model classes */}
            {health && (
              <div className="rounded-2xl border border-border bg-card p-5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
                  Model Classes ({health.classes.length})
                </h3>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {health.classes.map((cls) => (
                    <span
                      key={cls}
                      className="rounded-md bg-background px-2 py-0.5 text-[11px] font-medium"
                    >
                      {cls}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

function Stat({
  value,
  unit,
  label,
}: {
  value: string;
  unit?: string;
  label: string;
}) {
  return (
    <div className="rounded-xl bg-background p-3 text-center">
      <p className="text-xl font-bold text-accent">
        {value}
        {unit && <span className="ml-0.5 text-xs font-normal text-muted">{unit}</span>}
      </p>
      <p className="mt-0.5 text-[10px] text-muted">{label}</p>
    </div>
  );
}
