import { useEffect, useRef, useState } from "react";

const API = "/api";

type Scores = Record<string, number>;

interface Prediction {
  label: string;
  scores: Scores;
  confidence: number;
  inference_ms: number;
  class_names: string[];
}

interface Sample {
  name: string;
  truth: string;
  url: string;
}

const REAL = "Real";
const AI = "AI-Generated";

function pct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

function LogoMark() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2l8 4v6c0 5-3.5 8-8 10-4.5-2-8-5-8-10V6l8-4z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  );
}

function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewName, setPreviewName] = useState<string>("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const objectUrl = useRef<string | null>(null);

  useEffect(() => {
    fetch(`${API}/samples`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r)))
      .then((d) => setSamples(d.samples ?? []))
      .catch(() => {});
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  function reset() {
    setResult(null);
    setError(null);
  }

  async function runPrediction(promise: Promise<Response>) {
    setLoading(true);
    reset();
    try {
      const res = await promise;
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Request failed (${res.status})`);
      }
      const data: Prediction = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  function handleFile(file: File) {
    if (!file.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    const url = URL.createObjectURL(file);
    objectUrl.current = url;
    setPreviewUrl(url);
    setPreviewName(file.name);
    const fd = new FormData();
    fd.append("file", file);
    runPrediction(fetch(`${API}/predict`, { method: "POST", body: fd }));
  }

  function handleSample(s: Sample) {
    if (objectUrl.current) {
      URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = null;
    }
    setPreviewUrl(`/${s.url}`);
    setPreviewName(s.name);
    runPrediction(fetch(`${API}/predict-sample/${encodeURIComponent(s.name)}`, { method: "POST" }));
  }

  function clearImage() {
    if (objectUrl.current) {
      URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = null;
    }
    setPreviewUrl(null);
    setPreviewName("");
    reset();
  }

  const realScore = result?.scores?.[REAL] ?? 0;
  const aiScore = result?.scores?.[AI] ?? 0;

  return (
    <div className="app-bg">
      <div className="shell">
        <header className="topbar">
          <div className="brand">
            <div className="logo">
              <LogoMark />
            </div>
            <div>
              <h1>DeepFake Detector</h1>
              <p>cnn · 128×128 · sigmoid → P(ai-generated)</p>
            </div>
          </div>
          <div className="status-pill">
            <span className="dot" />
            model ready
          </div>
        </header>

        <div className="grid">
          {/* Input panel */}
          <section className="panel">
            <div className="panel-title">
              <span className="idx">01</span> input image
            </div>

            {previewUrl ? (
              <div className="preview-wrap">
                <img src={previewUrl} alt={previewName} />
                <div className="preview-actions">
                  <button className="icon-btn" onClick={() => fileInput.current?.click()}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                    replace
                  </button>
                  <button className="icon-btn" onClick={clearImage}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                    clear
                  </button>
                </div>
              </div>
            ) : (
              <div
                className={`dropzone${dragging ? " drag" : ""}`}
                onClick={() => fileInput.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  const f = e.dataTransfer.files?.[0];
                  if (f) handleFile(f);
                }}
              >
                <div>
                  <svg className="dz-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <circle cx="9" cy="9" r="2" />
                    <path d="M21 15l-5-5L5 21" />
                  </svg>
                  <h3>Drop an image here</h3>
                  <p>or click to browse · PNG / JPG</p>
                  <div className="dz-hint">resized to 128×128 · normalized /255</div>
                </div>
              </div>
            )}

            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
                e.target.value = "";
              }}
            />

            <div className="panel-title" style={{ marginTop: 24 }}>
              <span className="idx">02</span> sample library
            </div>
            <div className="samples">
              {samples.map((s) => (
                <button key={s.name} className="thumb" onClick={() => handleSample(s)} title={s.name}>
                  <img src={`/${s.url}`} alt={s.name} loading="lazy" />
                  <span className={`tag ${s.truth === AI ? "ai" : "real"}`}>{s.truth === AI ? "AI" : "REAL"}</span>
                </button>
              ))}
            </div>
          </section>

          {/* Result panel */}
          <section className="panel">
            <div className="panel-title">
              <span className="idx">03</span> analysis
            </div>

            {loading ? (
              <div className="loading">
                <div>
                  <div className="spinner" />
                  <p>running inference</p>
                  <p className="sub">first run loads the model — this can take a moment</p>
                </div>
              </div>
            ) : result ? (
              <div className="fade-in">
                <div className={`verdict ${result.label === AI ? "ai" : "real"}`}>
                  <div>
                    <div className="verdict-label">prediction</div>
                    <div className="verdict-value">{result.label}</div>
                  </div>
                  <div className="confidence-ring">
                    <div className="num">{pct(result.confidence)}</div>
                    <div className="lbl">confidence</div>
                  </div>
                </div>

                <div className="bars">
                  <div className="bar-row">
                    <div className="bar-head">
                      <span className="bar-name">{REAL}</span>
                      <span className="bar-pct">{pct(realScore)}</span>
                    </div>
                    <div className="bar-track">
                      <div className="bar-fill real" style={{ width: pct(realScore) }} />
                    </div>
                  </div>
                  <div className="bar-row">
                    <div className="bar-head">
                      <span className="bar-name">{AI}</span>
                      <span className="bar-pct">{pct(aiScore)}</span>
                    </div>
                    <div className="bar-track">
                      <div className="bar-fill ai" style={{ width: pct(aiScore) }} />
                    </div>
                  </div>
                </div>

                <div className="meta-row">
                  <div className="meta-item">
                    <div className="meta-k">inference</div>
                    <div className="meta-v">{result.inference_ms.toFixed(0)} ms</div>
                  </div>
                  <div className="meta-item">
                    <div className="meta-k">raw P(ai)</div>
                    <div className="meta-v">{aiScore.toFixed(4)}</div>
                  </div>
                  <div className="meta-item">
                    <div className="meta-k">threshold</div>
                    <div className="meta-v">0.50</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="result-empty">
                <div>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="7" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  </svg>
                  <p>upload or pick a sample to analyze</p>
                </div>
              </div>
            )}

            {error && <div className="error-box">{error}</div>}
          </section>
        </div>

        <div className="footnote">
          Predictions are probabilistic and may be wrong. This is a demonstration model, not a forensic tool.
        </div>
      </div>
    </div>
  );
}

export default App;
