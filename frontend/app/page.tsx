"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ──

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
  cropped_images: string[];
  model_classes?: string[];
}

interface HealthInfo {
  status: string;
  model_loaded: boolean;
  model_path: string | null;
  classes: string[];
}

interface FurnitureAnalysis {
  type: string;
  material: string;
  color: string;
  style: string;
  approximate_dimensions: string;
  condition_assessment: string;
  brand_guess: string | null;
  search_keywords: string;
  description: string;
}

interface SearchListing {
  title: string;
  price: string | number;
  store: string;
  link: string;
  thumbnail: string;
  source?: string;
}

interface ItemState {
  detection: Detection;
  cropImage: string;
  analysis: FurnitureAnalysis | null;
  analyzingStatus: "idle" | "loading" | "done" | "error";
  analyzeError?: string;
  exactResults: SearchListing[];
  altResults: SearchListing[];
  searchingExact: boolean;
  searchingAlt: boolean;
}

const PAKISTAN_CITIES = [
  "Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad",
  "Multan", "Peshawar", "Quetta", "Sialkot", "Gujranwala",
  "Hyderabad", "Bahawalpur", "Sargodha", "Abbottabad", "Mardan",
];

// ── Page ──

type AppView = "upload" | "results" | "report";

export default function Home() {
  const [view, setView] = useState<AppView>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.5);
  const [dragActive, setDragActive] = useState(false);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [items, setItems] = useState<ItemState[]>([]);
  const [city, setCity] = useState("Lahore");
  const [cityQuery, setCityQuery] = useState("Lahore");
  const [showCityDropdown, setShowCityDropdown] = useState(false);

  const [reportMd, setReportMd] = useState<string | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/`).then((r) => r.json()).then(setHealth).catch(() => setHealth(null));
  }, []);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) { setError("Please upload an image file"); return; }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setItems([]);
    setReportMd(null);
    setView("upload");
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const runDetection = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", selectedFile);
      const res = await fetch(`${API_URL}/detect/full?confidence=${confidence}`, { method: "POST", body: fd });
      if (!res.ok) { const e = await res.json().catch(() => null); throw new Error(e?.detail || `Error ${res.status}`); }
      const data: DetectionResult = await res.json();
      setResult(data);
      setItems(data.detections.map((det, i) => ({
        detection: det, cropImage: data.cropped_images[i] || "",
        analysis: null, analyzingStatus: "idle", exactResults: [], altResults: [],
        searchingExact: false, searchingAlt: false,
      })));
      if (data.count > 0) setView("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed");
    } finally { setLoading(false); }
  };

  const reset = () => {
    setSelectedFile(null); setPreviewUrl(null); setResult(null);
    setError(null); setItems([]); setReportMd(null); setView("upload");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const updateItem = (idx: number, patch: Partial<ItemState>) =>
    setItems((prev) => prev.map((it, i) => (i === idx ? { ...it, ...patch } : it)));

  const analyzeItem = async (index: number) => {
    const item = items[index];
    if (!item.cropImage) return;
    updateItem(index, { analyzingStatus: "loading", analyzeError: undefined });
    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crop_image: item.cropImage }),
      });
      if (!res.ok) { const e = await res.json().catch(() => null); throw new Error(e?.detail || `Error ${res.status}`); }
      const data = await res.json();
      updateItem(index, { analysis: data.analysis, analyzingStatus: "done" });
    } catch (err) {
      updateItem(index, { analyzingStatus: "error", analyzeError: err instanceof Error ? err.message : "Analysis failed" });
    }
  };

  const searchItem = async (index: number, mode: "exact" | "alternative") => {
    const item = items[index];
    let analysis = item.analysis;
    if (!analysis) {
      updateItem(index, { analyzingStatus: "loading", analyzeError: undefined });
      try {
        const res = await fetch(`${API_URL}/analyze`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ crop_image: item.cropImage }),
        });
        if (!res.ok) { const e = await res.json().catch(() => null); throw new Error(e?.detail || `Error ${res.status}`); }
        const data = await res.json();
        analysis = data.analysis;
        updateItem(index, { analysis, analyzingStatus: "done" });
      } catch (err) {
        updateItem(index, { analyzingStatus: "error", analyzeError: err instanceof Error ? err.message : "Analysis failed" });
        return;
      }
    }
    const isExact = mode === "exact";
    updateItem(index, isExact ? { searchingExact: true } : { searchingAlt: true });
    try {
      const res = await fetch(`${API_URL}/search`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysis, city, mode }),
      });
      if (!res.ok) throw new Error("Search failed");
      const data = await res.json();
      updateItem(index, isExact ? { exactResults: data.results, searchingExact: false } : { altResults: data.results, searchingAlt: false });
    } catch { updateItem(index, isExact ? { searchingExact: false } : { searchingAlt: false }); }
  };

  const generateReport = async () => {
    setGeneratingReport(true);
    try {
      const payload = items.map((it) => ({
        detection: { class_name: it.detection.class_name, confidence: it.detection.confidence, furniture_type: it.detection.furniture_type, condition: it.detection.condition },
        analysis: it.analysis,
        search_results: { exact: it.exactResults, alternative: it.altResults },
        city,
      }));
      const res = await fetch(`${API_URL}/report`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items: payload }),
      });
      if (!res.ok) throw new Error("Report generation failed");
      const data = await res.json();
      setReportMd(data.report);
      setView("report");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Report generation failed");
    } finally { setGeneratingReport(false); }
  };

  const filteredCities = PAKISTAN_CITIES.filter((c) => c.toLowerCase().includes(cityQuery.toLowerCase()));
  const backendOnline = health?.status === "ok";

  return (
    <div className="min-h-screen bg-background bg-grid">
      <div className="bg-radial-glow fixed inset-0 pointer-events-none" />

      <Header backendOnline={backendOnline} modelPath={health?.model_path || null} />

      <div className="relative">
        {view === "upload" && (
          <UploadView
            {...{ previewUrl, result, loading, error, confidence, dragActive, backendOnline, health, fileInputRef, items }}
            onSetConfidence={setConfidence}
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onFileSelect={() => fileInputRef.current?.click()}
            onFileChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
            onDetect={runDetection}
            onReset={reset}
            onViewResults={() => setView("results")}
          />
        )}
        {view === "results" && (
          <ResultsView
            {...{ result, items, city, cityQuery, showCityDropdown, filteredCities, generatingReport }}
            onCityQueryChange={setCityQuery}
            onCitySelect={(c) => { setCity(c); setCityQuery(c); setShowCityDropdown(false); }}
            onShowDropdown={() => setShowCityDropdown(true)}
            onHideDropdown={() => setTimeout(() => setShowCityDropdown(false), 150)}
            onAnalyze={analyzeItem}
            onSearch={searchItem}
            onGenerateReport={generateReport}
            onBack={() => setView("upload")}
          />
        )}
        {view === "report" && (
          <ReportView reportMd={reportMd} onBack={() => setView("results")} onPrint={() => window.print()} />
        )}
      </div>
    </div>
  );
}

// ── Header ──

function Header({ backendOnline, modelPath }: { backendOnline: boolean; modelPath: string | null }) {
  return (
    <header className="sticky top-0 z-50 glass-strong print:hidden">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-cyan-500 text-white font-bold text-lg glow-sm">
            V
            <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-accent to-cyan-500 opacity-50 blur-md -z-10" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight gradient-text">VisionAI</h1>
            <p className="text-[10px] leading-none text-muted tracking-widest uppercase">
              Intelligent Furniture Detection
            </p>
          </div>
        </div>
        <div className="flex items-center gap-5 text-xs">
          <span className="flex items-center gap-2">
            <span className={`relative inline-block h-2 w-2 rounded-full ${backendOnline ? "bg-success" : "bg-danger"}`}>
              {backendOnline && <span className="absolute inset-0 rounded-full bg-success animate-ping opacity-40" />}
            </span>
            <span className="text-muted">{backendOnline ? "System Online" : "System Offline"}</span>
          </span>
          {modelPath && (
            <span className="hidden sm:flex items-center gap-2 rounded-full glass px-3 py-1 text-[10px] font-mono text-muted">
              <IconCpu />
              {modelPath.split(/[\\/]/).pop()}
            </span>
          )}
        </div>
      </div>
    </header>
  );
}

// ── Upload View ──

function UploadView({
  previewUrl, result, loading, error, confidence, dragActive,
  backendOnline, health, fileInputRef, items,
  onSetConfidence, onDrop, onDragOver, onDragLeave, onFileSelect, onFileChange,
  onDetect, onReset, onViewResults,
}: {
  previewUrl: string | null; result: DetectionResult | null; loading: boolean;
  error: string | null; confidence: number; dragActive: boolean; backendOnline: boolean;
  health: HealthInfo | null; fileInputRef: React.RefObject<HTMLInputElement | null>;
  items: ItemState[];
  onSetConfidence: (v: number) => void; onDrop: (e: React.DragEvent) => void;
  onDragOver: (e: React.DragEvent) => void; onDragLeave: () => void;
  onFileSelect: () => void; onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onDetect: () => void; onReset: () => void; onViewResults: () => void;
}) {
  const grouped = result?.detections.reduce(
    (acc, d) => { const k = d.class_name; if (!acc[k]) acc[k] = []; acc[k].push(d); return acc; },
    {} as Record<string, Detection[]>
  );

  return (
    <main className="mx-auto max-w-7xl px-6 py-10 animate-fade-up">
      {!previewUrl && (
        <div className="mb-10 text-center">
          <h2 className="text-4xl font-extrabold tracking-tight">
            <span className="gradient-text">AI-Powered</span> Furniture Detection
          </h2>
          <p className="mt-3 text-muted max-w-lg mx-auto">
            Upload an image to detect furniture, assess condition, search marketplaces, and generate professional reports — all powered by AI.
          </p>
        </div>
      )}

      <div className="grid gap-8 lg:grid-cols-3">
        <div className="space-y-5 lg:col-span-2">
          <div
            className={`relative rounded-2xl transition-all duration-300 overflow-hidden ${
              dragActive ? "glow scale-[1.01]" : ""
            } ${!previewUrl ? "p-16" : "p-4"} gradient-border glass`}
            onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
          >
            {!previewUrl ? (
              <div className="flex flex-col items-center gap-5 text-center">
                <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-accent/20 to-cyan-500/20 animate-float">
                  <IconUpload className="h-10 w-10 text-accent" />
                  <div className="absolute inset-0 rounded-2xl border border-accent/30 animate-pulse-glow" />
                </div>
                <div>
                  <p className="text-xl font-bold">
                    Drop your image here, or{" "}
                    <button onClick={onFileSelect} className="gradient-text font-bold hover:opacity-80 transition-opacity">
                      browse files
                    </button>
                  </p>
                  <p className="mt-2 text-sm text-muted">Supports JPEG, PNG, WebP — any resolution</p>
                </div>
                <div className="flex gap-3 text-[10px] text-muted">
                  {["YOLOv11 Detection", "Gemini Vision AI", "Smart Search"].map((t) => (
                    <span key={t} className="flex items-center gap-1.5 rounded-full glass px-3 py-1.5">
                      <span className="h-1 w-1 rounded-full bg-accent" />
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative overflow-hidden rounded-xl bg-black/30 ring-1 ring-white/5">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={result?.annotated_image || previewUrl}
                    alt="Preview"
                    className="mx-auto max-h-[520px] w-full object-contain"
                  />
                  {loading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-background/70 backdrop-blur-md">
                      <div className="flex flex-col items-center gap-4">
                        <div className="relative h-14 w-14">
                          <div className="absolute inset-0 rounded-full border-2 border-accent/20" />
                          <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-accent animate-spin" />
                          <div className="absolute inset-2 rounded-full border-2 border-transparent border-b-cyan-400 animate-spin" style={{ animationDirection: "reverse", animationDuration: "0.8s" }} />
                        </div>
                        <div className="text-center">
                          <p className="text-sm font-semibold gradient-text">Analyzing Image</p>
                          <p className="text-[10px] text-muted mt-1">Running YOLOv11 inference...</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <button onClick={onDetect} disabled={loading || !backendOnline}
                    className="flex-1 rounded-xl bg-gradient-to-r from-accent to-cyan-500 px-6 py-3 text-sm font-bold text-white transition-all hover:opacity-90 hover:shadow-lg hover:shadow-accent/20 disabled:opacity-40 disabled:cursor-not-allowed">
                    {loading ? "Processing..." : result ? "Re-run Detection" : "Run Detection"}
                  </button>
                  {result && items.length > 0 && (
                    <button onClick={onViewResults}
                      className="rounded-xl bg-success/20 border border-success/30 text-success px-5 py-3 text-sm font-bold transition-all hover:bg-success/30">
                      View Items ({items.length})
                    </button>
                  )}
                  <button onClick={onReset}
                    className="rounded-xl glass px-5 py-3 text-sm font-medium transition-colors hover:bg-white/5">
                    Clear
                  </button>
                </div>
              </div>
            )}
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={onFileChange} />
          </div>

          {error && (
            <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger flex items-center gap-2">
              <IconAlert /> {error}
            </div>
          )}
          {!backendOnline && (
            <div className="rounded-xl border border-warning/30 bg-warning/10 px-4 py-3 text-sm text-warning flex items-center gap-2">
              <IconAlert /> Backend offline — run: <code className="ml-1 rounded bg-warning/20 px-2 py-0.5 font-mono text-xs">uvicorn backend.main:app --reload</code>
            </div>
          )}
        </div>

        {/* Right sidebar */}
        <div className="space-y-5">
          <div className="glass rounded-2xl p-5">
            <SectionLabel text="Settings" icon={<IconSlider />} />
            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted">Confidence</span>
                <span className="font-mono font-bold gradient-text">{(confidence * 100).toFixed(0)}%</span>
              </div>
              <input type="range" min="0.1" max="1" step="0.05" value={confidence}
                onChange={(e) => onSetConfidence(parseFloat(e.target.value))}
                className="w-full accent-accent h-1.5 rounded-full appearance-none bg-surface cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:shadow-accent/30" />
            </div>
          </div>

          {result && (
            <div className="glass rounded-2xl p-5 animate-fade-up">
              <SectionLabel text="Detection Summary" icon={<IconChart />} />
              <div className="mt-4 grid grid-cols-2 gap-3">
                <StatCard value={String(result.count)} label="Objects" icon={<IconBox />} />
                <StatCard value={`${result.inference_time_ms.toFixed(0)}`} unit="ms" label="Inference" icon={<IconZap />} />
                <StatCard value={`${result.image_width}x${result.image_height}`} label="Resolution" icon={<IconImage />} />
                <StatCard value={String(Object.keys(grouped || {}).length)} label="Classes" icon={<IconTag />} />
              </div>
            </div>
          )}

          {result && grouped && result.count > 0 && (
            <div className="glass rounded-2xl p-5 animate-fade-up" style={{ animationDelay: "0.1s" }}>
              <SectionLabel text="Detections" icon={<IconList />} />
              <div className="mt-4 max-h-[350px] space-y-3 overflow-y-auto pr-1">
                {Object.entries(grouped).map(([cls, dets]) => (
                  <div key={cls}>
                    <p className="mb-1.5 text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-accent" />
                      {cls} <span className="text-accent">({dets.length})</span>
                    </p>
                    <div className="space-y-1.5">
                      {dets.map((d, i) => {
                        const cond = d.condition;
                        const color = cond === "broken" || cond === "damaged" ? "danger" : cond === "wornout" ? "warning" : "accent";
                        return (
                          <div key={i} className={`flex items-center justify-between rounded-lg border-l-2 border-${color} bg-${color}/5 px-3 py-2`}>
                            <span className="text-sm font-medium capitalize">{cond || cls}</span>
                            <span className="font-mono text-xs font-bold text-muted">{(d.confidence * 100).toFixed(1)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {health && (
            <div className="glass rounded-2xl p-5 animate-fade-up" style={{ animationDelay: "0.2s" }}>
              <SectionLabel text={`Model Classes (${health.classes.length})`} icon={<IconCpu />} />
              <div className="mt-3 flex flex-wrap gap-1.5">
                {health.classes.map((cls) => (
                  <span key={cls} className="rounded-md bg-surface px-2 py-0.5 text-[10px] font-medium text-muted border border-border">
                    {cls}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

// ── Results View ──

function ResultsView({
  result, items, city, cityQuery, showCityDropdown, filteredCities,
  generatingReport, onCityQueryChange, onCitySelect, onShowDropdown,
  onHideDropdown, onAnalyze, onSearch, onGenerateReport, onBack,
}: {
  result: DetectionResult | null; items: ItemState[]; city: string;
  cityQuery: string; showCityDropdown: boolean; filteredCities: string[];
  generatingReport: boolean;
  onCityQueryChange: (v: string) => void; onCitySelect: (c: string) => void;
  onShowDropdown: () => void; onHideDropdown: () => void;
  onAnalyze: (i: number) => void; onSearch: (i: number, m: "exact" | "alternative") => void;
  onGenerateReport: () => void; onBack: () => void;
}) {
  return (
    <main className="mx-auto max-w-7xl px-6 py-8 animate-fade-up">
      {/* Toolbar */}
      <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <button onClick={onBack} className="flex items-center gap-2 rounded-xl glass px-4 py-2.5 text-sm font-medium hover:bg-white/5 transition-colors">
            <IconBack /> Back
          </button>
          <div>
            <h2 className="text-2xl font-extrabold tracking-tight">
              Detected Items <span className="gradient-text">{items.length}</span>
            </h2>
            <p className="text-xs text-muted mt-0.5">Click items below to analyze and search marketplaces</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <label className="block text-[9px] font-bold uppercase tracking-widest text-muted mb-1 flex items-center gap-1">
              <IconMap /> Search City
            </label>
            <input type="text" value={cityQuery}
              onChange={(e) => { onCityQueryChange(e.target.value); onShowDropdown(); }}
              onFocus={onShowDropdown} onBlur={onHideDropdown}
              className="w-44 rounded-xl glass px-3 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-accent/50 placeholder-muted/50"
              placeholder="Select city..." />
            {showCityDropdown && filteredCities.length > 0 && (
              <div className="absolute top-full left-0 z-50 mt-1 w-44 overflow-hidden rounded-xl glass-strong shadow-2xl shadow-black/50">
                {filteredCities.map((c) => (
                  <button key={c} onMouseDown={() => onCitySelect(c)}
                    className={`block w-full px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent/10 ${c === city ? "bg-accent/10 text-accent font-semibold" : "text-foreground"}`}>
                    {c}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="self-end">
            <button onClick={onGenerateReport}
              disabled={generatingReport || items.every((it) => !it.analysis)}
              className="rounded-xl bg-gradient-to-r from-accent to-cyan-500 px-5 py-2.5 text-sm font-bold text-white transition-all hover:opacity-90 hover:shadow-lg hover:shadow-accent/20 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2">
              {generatingReport ? <Spinner /> : <IconReport />}
              {generatingReport ? "Generating..." : "Generate Report"}
            </button>
          </div>
        </div>
      </div>

      {/* Annotated preview */}
      {result?.annotated_image && (
        <div className="mb-8 overflow-hidden rounded-2xl glass ring-1 ring-white/5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={result.annotated_image} alt="Annotated" className="mx-auto max-h-[380px] w-full object-contain" />
        </div>
      )}

      {/* Item grid */}
      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
        {items.map((item, idx) => (
          <ItemCard key={idx} item={item} index={idx} city={city}
            onAnalyze={() => onAnalyze(idx)}
            onSearchExact={() => onSearch(idx, "exact")}
            onSearchAlt={() => onSearch(idx, "alternative")} />
        ))}
      </div>
    </main>
  );
}

// ── Item Card ──

function ItemCard({ item, index, city, onAnalyze, onSearchExact, onSearchAlt }: {
  item: ItemState; index: number; city: string;
  onAnalyze: () => void; onSearchExact: () => void; onSearchAlt: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const cond = item.detection.condition;
  const borderColor = cond === "broken" || cond === "damaged" ? "border-danger/40" : cond === "wornout" ? "border-warning/40" : "border-border-strong";

  return (
    <div className={`glass rounded-2xl overflow-hidden transition-all hover:shadow-xl hover:shadow-accent/5 ${borderColor} border animate-fade-up`}
      style={{ animationDelay: `${index * 0.05}s` }}>
      {/* Header */}
      <div className="flex gap-4 p-4">
        {item.cropImage && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={item.cropImage} alt={item.detection.class_name}
            className="h-24 w-24 rounded-xl object-cover ring-1 ring-white/10 flex-shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="font-bold text-sm capitalize">{item.detection.class_name}</h3>
              {cond && (
                <span className={`inline-block mt-1.5 rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                  cond === "broken" || cond === "damaged" ? "bg-danger/15 text-danger border border-danger/30" :
                  cond === "wornout" ? "bg-warning/15 text-warning border border-warning/30" :
                  "bg-success/15 text-success border border-success/30"
                }`}>{cond}</span>
              )}
            </div>
            <span className="font-mono text-xs font-bold text-accent bg-accent/10 rounded-full px-2 py-0.5">
              {(item.detection.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <p className="mt-1.5 text-[10px] text-muted font-mono">
            ITEM #{index + 1} &middot; {item.detection.furniture_type || item.detection.class_name}
          </p>
        </div>
      </div>

      {/* Analysis */}
      {item.analysis && (
        <div className="border-t border-border px-4 py-3 bg-surface/50">
          <button onClick={() => setExpanded(!expanded)}
            className="flex w-full items-center justify-between text-[10px] font-bold uppercase tracking-widest text-muted">
            <span className="flex items-center gap-1.5"><IconSpark /> AI Analysis</span>
            <svg className={`h-3.5 w-3.5 transition-transform duration-200 ${expanded ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {expanded && (
            <div className="mt-3 space-y-2 text-xs animate-fade-up">
              {[
                ["Type", item.analysis.type], ["Material", item.analysis.material],
                ["Color", item.analysis.color], ["Style", item.analysis.style],
                ["Size", item.analysis.approximate_dimensions], ["Condition", item.analysis.condition_assessment],
                ...(item.analysis.brand_guess ? [["Brand", item.analysis.brand_guess]] : []),
              ].map(([label, value]) => (
                <div key={label} className="flex justify-between gap-3">
                  <span className="text-muted">{label}</span>
                  <span className="text-right capitalize font-medium">{value}</span>
                </div>
              ))}
              <p className="mt-2 text-muted italic text-[11px] pt-2 border-t border-border">{item.analysis.description}</p>
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="border-t border-border p-4 space-y-3">
        {item.analyzingStatus === "idle" && !item.analysis && (
          <button onClick={onAnalyze}
            className="w-full rounded-xl bg-accent/10 border border-accent/20 px-3 py-2.5 text-xs font-bold text-accent transition-all hover:bg-accent/20 hover:border-accent/30 flex items-center justify-center gap-2">
            <IconSpark /> Analyze with AI
          </button>
        )}
        {item.analyzingStatus === "loading" && (
          <div className="flex items-center justify-center gap-2 py-2.5 text-xs text-muted">
            <Spinner /> Analyzing with Gemini...
          </div>
        )}
        {item.analyzingStatus === "error" && (
          <div className="text-center py-2 space-y-1.5">
            <p className="text-xs text-danger">{item.analyzeError || "Analysis failed"}</p>
            <button onClick={onAnalyze} className="text-[10px] text-accent font-medium hover:underline">Retry Analysis</button>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2">
          <button onClick={onSearchExact} disabled={item.searchingExact}
            className="rounded-xl bg-gradient-to-r from-accent to-accent-light px-3 py-2.5 text-xs font-bold text-white transition-all hover:opacity-90 disabled:opacity-40 flex items-center justify-center gap-1.5">
            {item.searchingExact ? <Spinner /> : <IconSearch />} Find Exact
          </button>
          <button onClick={onSearchAlt} disabled={item.searchingAlt}
            className="rounded-xl glass border-accent/30 text-accent px-3 py-2.5 text-xs font-bold transition-all hover:bg-accent/10 disabled:opacity-40 flex items-center justify-center gap-1.5">
            {item.searchingAlt ? <Spinner /> : <IconShuffle />} Alternative
          </button>
        </div>
        <p className="text-[9px] text-center text-muted font-mono uppercase tracking-wider flex items-center justify-center gap-1">
          <IconMap className="h-3 w-3" /> {city}
        </p>
      </div>

      {/* Search Results */}
      {item.exactResults.length > 0 && <SearchResults title="Exact Matches" results={item.exactResults} accent />}
      {item.altResults.length > 0 && <SearchResults title="Alternatives" results={item.altResults} />}
    </div>
  );
}

// ── Search Results ──

function SearchResults({ title, results, accent }: { title: string; results: SearchListing[]; accent?: boolean }) {
  return (
    <div className="border-t border-border px-4 py-3">
      <h4 className={`text-[9px] font-bold uppercase tracking-widest mb-3 flex items-center gap-1.5 ${accent ? "gradient-text" : "text-muted"}`}>
        <IconSearch className="h-3 w-3" /> {title} ({results.length})
      </h4>
      <div className="space-y-2 max-h-56 overflow-y-auto">
        {results.map((r, i) => (
          <div key={i} className="flex items-start gap-3 rounded-xl bg-surface/50 p-2.5 border border-border transition-colors hover:border-accent/20">
            {r.thumbnail && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={r.thumbnail} alt="" className="h-11 w-11 rounded-lg object-cover flex-shrink-0 ring-1 ring-white/10" />
            )}
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium leading-tight line-clamp-2">
                {r.link ? <a href={r.link} target="_blank" rel="noopener noreferrer" className="hover:text-accent transition-colors">{r.title}</a> : r.title}
              </p>
              <div className="mt-1 flex items-center gap-2 text-[10px] text-muted">
                {r.price && <span className="font-bold text-success">{typeof r.price === "number" ? `$${r.price}` : r.price}</span>}
                {r.store && <span>&middot; {r.store}</span>}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Report View ──

function ReportView({ reportMd, onBack, onPrint }: { reportMd: string | null; onBack: () => void; onPrint: () => void }) {
  return (
    <main className="mx-auto max-w-4xl px-6 py-8 animate-fade-up">
      <div className="mb-6 flex items-center justify-between print:hidden">
        <button onClick={onBack} className="flex items-center gap-2 rounded-xl glass px-4 py-2.5 text-sm font-medium hover:bg-white/5 transition-colors">
          <IconBack /> Back to Items
        </button>
        <button onClick={onPrint}
          className="rounded-xl bg-gradient-to-r from-accent to-cyan-500 px-5 py-2.5 text-sm font-bold text-white transition-all hover:opacity-90 flex items-center gap-2">
          <IconPrint /> Print / Save PDF
        </button>
      </div>
      <div className="glass rounded-2xl p-8 print:bg-white print:text-black print:border-gray-200">
        <div className="prose prose-sm prose-invert max-w-none print:prose-neutral">
          <MarkdownRenderer content={reportMd || ""} />
        </div>
      </div>
    </main>
  );
}

// ── Markdown ──

function MarkdownRenderer({ content }: { content: string }) {
  return <div dangerouslySetInnerHTML={{ __html: mdToHtml(content) }} />;
}

function mdToHtml(md: string): string {
  let h = md;
  h = h.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  h = h.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  h = h.replace(/^# (.+)$/gm, "<h1>$1</h1>");
  h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/\*(.+?)\*/g, "<em>$1</em>");
  h = h.replace(/`(.+?)`/g, "<code>$1</code>");
  h = h.replace(/^\|(.+)\|$/gm, (m) => {
    const cells = m.split("|").filter(Boolean).map((c) => c.trim());
    if (cells.every((c) => /^[-:]+$/.test(c))) return "<!--sep-->";
    return `<tr>${cells.map((c) => `<td class="border border-border px-3 py-1.5 text-xs">${c}</td>`).join("")}</tr>`;
  });
  h = h.replace(/((<tr>.*?<\/tr>\n?)+)/g, (block) =>
    `<table class="w-full border-collapse border border-border text-sm">${block.replace(/<!--sep-->\n?/g, "")}</table>`
  );
  h = h.replace(/^- (.+)$/gm, "<li>$1</li>");
  h = h.replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);
  h = h.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");
  h = h.replace(/^(?!<[hultdp]|<!--)(.+)$/gm, "<p>$1</p>");
  return h;
}

// ── Utility Components ──

function SectionLabel({ text, icon }: { text: string; icon?: React.ReactNode }) {
  return (
    <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-1.5">
      {icon} {text}
    </h3>
  );
}

function StatCard({ value, unit, label, icon }: { value: string; unit?: string; label: string; icon?: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-surface/50 border border-border p-3 text-center">
      <div className="flex items-center justify-center gap-1 text-accent mb-1 opacity-60">{icon}</div>
      <p className="text-lg font-extrabold gradient-text">
        {value}{unit && <span className="ml-0.5 text-[10px] font-normal text-muted">{unit}</span>}
      </p>
      <p className="mt-0.5 text-[9px] text-muted uppercase tracking-wider">{label}</p>
    </div>
  );
}

function Spinner() {
  return <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current/20 border-t-current" />;
}

// ── Icons (inline SVGs) ──

function IconUpload({ className = "h-5 w-5" }: { className?: string }) {
  return <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" /></svg>;
}
function IconBack() {
  return <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" /></svg>;
}
function IconSearch({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><circle cx="11" cy="11" r="8" /><path strokeLinecap="round" d="m21 21-4.35-4.35" /></svg>;
}
function IconSpark() {
  return <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L9.19 9.19 2 12l7.19 2.81L12 22l2.81-7.19L22 12l-7.19-2.81L12 2z" /></svg>;
}
function IconShuffle() {
  return <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" /></svg>;
}
function IconReport() {
  return <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" /></svg>;
}
function IconPrint() {
  return <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M6.72 13.829c-.24.03-.48.062-.72.096m.72-.096a42.415 42.415 0 0 1 10.56 0m-10.56 0L6.34 18m10.94-4.171c.24.03.48.062.72.096m-.72-.096L17.66 18m0 0 .229 2.523a1.125 1.125 0 0 1-1.12 1.227H7.231c-.662 0-1.18-.568-1.12-1.227L6.34 18m11.318 0h1.091A2.25 2.25 0 0 0 21 15.75V9.456c0-1.081-.768-2.015-1.837-2.175a48.055 48.055 0 0 0-1.913-.247M6.34 18H5.25A2.25 2.25 0 0 1 3 15.75V9.456c0-1.081.768-2.015 1.837-2.175a48.041 48.041 0 0 1 1.913-.247m0 0a48.159 48.159 0 0 1 12.5 0m-12.5 0V6.75A2.25 2.25 0 0 1 7.5 4.5h9a2.25 2.25 0 0 1 2.25 2.25v2.956" /></svg>;
}
function IconMap({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" /><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1 1 15 0Z" /></svg>;
}
function IconAlert() {
  return <svg className="h-4 w-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" /></svg>;
}
function IconSlider() {
  return <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75" /></svg>;
}
function IconChart() {
  return <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" /></svg>;
}
function IconBox() {
  return <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="m21 7.5-9-5.25L3 7.5m18 0-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25" /></svg>;
}
function IconZap() {
  return <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="m3.75 13.5 10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z" /></svg>;
}
function IconImage() {
  return <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0 0 22.5 18.75V5.25A2.25 2.25 0 0 0 20.25 3H3.75A2.25 2.25 0 0 0 1.5 5.25v13.5A2.25 2.25 0 0 0 3.75 21Z" /></svg>;
}
function IconTag() {
  return <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9.568 3H5.25A2.25 2.25 0 0 0 3 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 0 0 5.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 0 0 9.568 3Z" /><path strokeLinecap="round" strokeLinejoin="round" d="M6 6h.008v.008H6V6Z" /></svg>;
}
function IconList() {
  return <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 6.75h12M8.25 12h12m-12 5.25h12M3.75 6.75h.007v.008H3.75V6.75Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0ZM3.75 12h.007v.008H3.75V12Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm-.375 5.25h.007v.008H3.75v-.008Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z" /></svg>;
}
function IconCpu() {
  return <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 0 0 2.25-2.25V6.75a2.25 2.25 0 0 0-2.25-2.25H6.75A2.25 2.25 0 0 0 4.5 6.75v10.5a2.25 2.25 0 0 0 2.25 2.25Zm.75-12h9v9h-9v-9Z" /></svg>;
}
