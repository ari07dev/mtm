import { useState, useEffect, useRef } from "react";

// ─── DESIGN TOKENS ───────────────────────────────────────────────────────────
// Industrial/utilitarian aesthetic: dark steel, amber accents, monospace code feel
const COLORS = {
  bg: "#0a0c0f",
  surface: "#111418",
  panel: "#161b22",
  border: "#1e2530",
  accent: "#f59e0b",
  accentDim: "#92400e",
  green: "#22c55e",
  red: "#ef4444",
  blue: "#3b82f6",
  muted: "#4b5563",
  text: "#e2e8f0",
  textDim: "#64748b",
};

// ─── MTM CODE MAPPING ────────────────────────────────────────────────────────
const MTM_CODES = {
  walk_short: "WALK 1-4 STEPS",
  walk_medium: "WALK 5-7 STEPS",
  walk_long: "WALK 8-10 STEPS",
  walk_vlong: "WALK 11-15 STEPS",
  reach_get: "GET + HOLD OBJECT",
  grasp_hold: "GRASP + HOLD OBJECT",
  grasp_place: "GRASP + PLACE OBJECT",
  hold_put: "HOLD + PUT OBJECT",
  hold_slide: "HOLD + SLIDE OBJECT (M3)",
  position: "PT",
  idle: "IDLE / TRANSITION",
};

// ─── SKELETON KEYPOINT NAMES (COCO 17-point) ─────────────────────────────────
const KEYPOINT_NAMES = [
  "nose","left_eye","right_eye","left_ear","right_ear",
  "left_shoulder","right_shoulder","left_elbow","right_elbow",
  "left_wrist","right_wrist","left_hip","right_hip",
  "left_knee","right_knee","left_ankle","right_ankle"
];

const SKELETON_CONNECTIONS = [
  [0,1],[0,2],[1,3],[2,4],
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16]
];

// ─── MOCK DATA GENERATOR ─────────────────────────────────────────────────────
function generateMockFrame(frameIdx) {
  const t = frameIdx * 0.08;
  const walkPhase = Math.sin(t * 2);
  const armPhase = Math.sin(t * 2 + Math.PI);

  // Base skeleton (normalized 0-1 coords)
  const kps = [
    // head
    { x: 0.5, y: 0.12, conf: 0.98 },           // nose
    { x: 0.47, y: 0.10, conf: 0.96 },           // left_eye
    { x: 0.53, y: 0.10, conf: 0.96 },           // right_eye
    { x: 0.45, y: 0.11, conf: 0.90 },           // left_ear
    { x: 0.55, y: 0.11, conf: 0.90 },           // right_ear
    // shoulders
    { x: 0.42, y: 0.26, conf: 0.97 },           // left_shoulder
    { x: 0.58, y: 0.26, conf: 0.97 },           // right_shoulder
    // elbows
    { x: 0.38, y: 0.40 + armPhase * 0.04, conf: 0.92 },
    { x: 0.62, y: 0.40 - armPhase * 0.04, conf: 0.92 },
    // wrists
    { x: 0.35, y: 0.54 + armPhase * 0.06, conf: 0.88 },
    { x: 0.65, y: 0.54 - armPhase * 0.06, conf: 0.88 },
    // hips
    { x: 0.44, y: 0.54, conf: 0.96 },
    { x: 0.56, y: 0.54, conf: 0.96 },
    // knees
    { x: 0.43, y: 0.72 + walkPhase * 0.04, conf: 0.91 },
    { x: 0.57, y: 0.72 - walkPhase * 0.04, conf: 0.91 },
    // ankles
    { x: 0.42, y: 0.90 + walkPhase * 0.05, conf: 0.87 },
    { x: 0.58, y: 0.90 - walkPhase * 0.05, conf: 0.87 },
  ];

  // Detect action from frame pattern
  const segment = Math.floor(frameIdx / 25) % 8;
  const actions = [
    "walk_vlong","reach_get","walk_long","hold_put",
    "grasp_place","hold_slide","walk_medium","grasp_hold"
  ];
  const action = actions[segment];

  return {
    frame_id: frameIdx,
    timestamp: (frameIdx / 30).toFixed(3),
    keypoints: kps,
    detected_persons: 1,
    action_raw: action,
    mtm_code: MTM_CODES[action],
    confidence: 0.75 + Math.random() * 0.22,
    bbox: { x: 0.3, y: 0.05, w: 0.4, h: 0.92 },
  };
}

// ─── SKELETON CANVAS ─────────────────────────────────────────────────────────
function SkeletonCanvas({ frame, width = 200, height = 280 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!frame || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    // Background grid
    ctx.strokeStyle = "#1e2530";
    ctx.lineWidth = 0.5;
    for (let i = 0; i < width; i += 20) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, height); ctx.stroke();
    }
    for (let j = 0; j < height; j += 20) {
      ctx.beginPath(); ctx.moveTo(0, j); ctx.lineTo(width, j); ctx.stroke();
    }

    const kps = frame.keypoints;
    const toX = (v) => v * width;
    const toY = (v) => v * height;

    // Draw connections
    SKELETON_CONNECTIONS.forEach(([a, b]) => {
      const ka = kps[a], kb = kps[b];
      if (ka.conf > 0.5 && kb.conf > 0.5) {
        const grad = ctx.createLinearGradient(toX(ka.x), toY(ka.y), toX(kb.x), toY(kb.y));
        grad.addColorStop(0, "#f59e0b88");
        grad.addColorStop(1, "#3b82f688");
        ctx.strokeStyle = grad;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(toX(ka.x), toY(ka.y));
        ctx.lineTo(toX(kb.x), toY(kb.y));
        ctx.stroke();
      }
    });

    // Draw keypoints
    kps.forEach((kp, i) => {
      if (kp.conf < 0.5) return;
      const isHead = i < 5;
      const isHand = i === 9 || i === 10;
      const isFoot = i === 15 || i === 16;

      ctx.beginPath();
      ctx.arc(toX(kp.x), toY(kp.y), isHead ? 5 : isHand || isFoot ? 4 : 3, 0, Math.PI * 2);
      ctx.fillStyle = isHead ? "#f59e0b" : isHand ? "#22c55e" : isFoot ? "#3b82f6" : "#e2e8f0";
      ctx.fill();
      ctx.strokeStyle = "#0a0c0f";
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Confidence halo on head
    const nose = kps[0];
    ctx.beginPath();
    ctx.arc(toX(nose.x), toY(nose.y), 12, 0, Math.PI * 2);
    ctx.strokeStyle = "#f59e0b44";
    ctx.lineWidth = 1;
    ctx.stroke();
  }, [frame, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ borderRadius: 4, border: `1px solid ${COLORS.border}` }}
    />
  );
}

// ─── CONFIDENCE BAR ───────────────────────────────────────────────────────────
function ConfBar({ value, label, color = COLORS.accent }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
        <span style={{ fontSize: 10, color: COLORS.textDim, fontFamily: "monospace" }}>{label}</span>
        <span style={{ fontSize: 10, color, fontFamily: "monospace" }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 4, background: COLORS.border, borderRadius: 2 }}>
        <div style={{
          height: "100%", width: `${value * 100}%`,
          background: color, borderRadius: 2,
          transition: "width 0.3s ease"
        }} />
      </div>
    </div>
  );
}

// ─── MTM SEQUENCE LOG ─────────────────────────────────────────────────────────
function MTMLog({ entries }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [entries]);

  return (
    <div style={{
      background: COLORS.bg, border: `1px solid ${COLORS.border}`,
      borderRadius: 6, padding: 12, height: 320, overflowY: "auto",
      fontFamily: "monospace", fontSize: 12,
    }}>
      <div style={{ color: COLORS.accent, marginBottom: 8, letterSpacing: 2, fontSize: 10 }}>
        ▸ MTM OUTPUT STREAM
      </div>
      {entries.map((e, i) => (
        <div key={i} style={{
          display: "flex", gap: 12, padding: "3px 0",
          borderBottom: `1px solid ${COLORS.border}11`,
          animation: i === entries.length - 1 ? "fadeIn 0.3s ease" : "none"
        }}>
          <span style={{ color: COLORS.textDim, minWidth: 55 }}>{e.timestamp}s</span>
          <span style={{ color: COLORS.accentDim }}>#{String(e.frame_id).padStart(4, "0")}</span>
          <span style={{
            color: e.mtm_code.includes("WALK") ? COLORS.blue :
                   e.mtm_code.includes("GRASP") ? COLORS.green :
                   e.mtm_code.includes("HOLD") ? COLORS.accent : COLORS.text,
            fontWeight: 600
          }}>{e.mtm_code}</span>
          <span style={{ color: COLORS.textDim, marginLeft: "auto" }}>
            {(e.confidence * 100).toFixed(0)}%
          </span>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}

// ─── PIPELINE STAGE INDICATOR ─────────────────────────────────────────────────
function PipelineStages({ activeStage }) {
  const stages = [
    { id: 0, icon: "🎬", label: "VIDEO INPUT", sub: "30fps decode" },
    { id: 1, icon: "🦴", label: "YOLOv8-POSE", sub: "17 keypoints" },
    { id: 2, icon: "📊", label: "SKELETON SEQ", sub: "time series" },
    { id: 3, icon: "🧠", label: "ST-GCN", sub: "action class." },
    { id: 4, icon: "⏱", label: "MS-TCN", sub: "smoothing" },
    { id: 5, icon: "📝", label: "MTM OUTPUT", sub: "LLM format" },
  ];

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 0, flexWrap: "wrap" }}>
      {stages.map((s, i) => (
        <div key={s.id} style={{ display: "flex", alignItems: "center" }}>
          <div style={{
            padding: "6px 10px", borderRadius: 4,
            background: activeStage === s.id ? COLORS.accent + "22" : COLORS.panel,
            border: `1px solid ${activeStage === s.id ? COLORS.accent : COLORS.border}`,
            transition: "all 0.3s ease",
          }}>
            <div style={{ fontSize: 14, textAlign: "center" }}>{s.icon}</div>
            <div style={{
              fontSize: 8, fontFamily: "monospace", letterSpacing: 1,
              color: activeStage === s.id ? COLORS.accent : COLORS.textDim,
              textAlign: "center", fontWeight: 700
            }}>{s.label}</div>
            <div style={{ fontSize: 7, color: COLORS.muted, textAlign: "center" }}>{s.sub}</div>
          </div>
          {i < stages.length - 1 && (
            <div style={{
              width: 16, height: 1,
              background: activeStage > s.id ? COLORS.accent : COLORS.border,
              transition: "background 0.5s ease"
            }} />
          )}
        </div>
      ))}
    </div>
  );
}

// ─── KEYPOINT TABLE ───────────────────────────────────────────────────────────
function KeypointTable({ frame }) {
  if (!frame) return null;
  const selected = [0, 5, 6, 9, 10, 11, 12, 15, 16]; // key joints only
  return (
    <div style={{ fontFamily: "monospace", fontSize: 10 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 2, marginBottom: 4 }}>
        {["KEYPOINT", "X", "Y", "CONF"].map(h => (
          <div key={h} style={{ color: COLORS.textDim, letterSpacing: 1 }}>{h}</div>
        ))}
      </div>
      {selected.map(i => {
        const kp = frame.keypoints[i];
        return (
          <div key={i} style={{
            display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr",
            gap: 2, padding: "2px 0",
            borderBottom: `1px solid ${COLORS.border}44`
          }}>
            <div style={{ color: COLORS.accent, fontSize: 9 }}>{KEYPOINT_NAMES[i].toUpperCase()}</div>
            <div style={{ color: COLORS.text }}>{kp.x.toFixed(3)}</div>
            <div style={{ color: COLORS.text }}>{kp.y.toFixed(3)}</div>
            <div style={{ color: kp.conf > 0.85 ? COLORS.green : kp.conf > 0.7 ? COLORS.accent : COLORS.red }}>
              {kp.conf.toFixed(2)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── STATS MINI CARD ──────────────────────────────────────────────────────────
function StatCard({ label, value, sub, color = COLORS.accent }) {
  return (
    <div style={{
      background: COLORS.panel, border: `1px solid ${COLORS.border}`,
      borderRadius: 6, padding: "10px 14px", flex: 1
    }}>
      <div style={{ fontSize: 9, color: COLORS.textDim, fontFamily: "monospace", letterSpacing: 1 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: "monospace", lineHeight: 1.2 }}>{value}</div>
      {sub && <div style={{ fontSize: 9, color: COLORS.muted, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function MTMPipeline() {
  const [running, setRunning] = useState(false);
  const [frameIdx, setFrameIdx] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [mtmLog, setMtmLog] = useState([]);
  const [activeStage, setActiveStage] = useState(0);
  const [fps, setFps] = useState(0);
  const [totalFrames] = useState(300);
  const [lastMtm, setLastMtm] = useState("");
  const intervalRef = useRef(null);
  const fpsRef = useRef({ count: 0, last: Date.now() });

  const processFrame = (idx) => {
    const frame = generateMockFrame(idx);
    setCurrentFrame(frame);

    // Stage cycling
    const stage = idx % 18;
    setActiveStage(stage < 3 ? 0 : stage < 6 ? 1 : stage < 9 ? 2 : stage < 12 ? 3 : stage < 15 ? 4 : 5);

    // Log new MTM entries (deduplicate consecutive same codes)
    setMtmLog(prev => {
      if (prev.length === 0 || prev[prev.length - 1].mtm_code !== frame.mtm_code) {
        setLastMtm(frame.mtm_code);
        return [...prev.slice(-60), frame];
      }
      return prev;
    });

    // FPS calc
    fpsRef.current.count++;
    const now = Date.now();
    if (now - fpsRef.current.last >= 1000) {
      setFps(fpsRef.current.count);
      fpsRef.current = { count: 0, last: now };
    }
  };

  const startPipeline = () => {
    setRunning(true);
    setMtmLog([]);
    setFrameIdx(0);
    intervalRef.current = setInterval(() => {
      setFrameIdx(prev => {
        const next = (prev + 1) % totalFrames;
        processFrame(next);
        return next;
      });
    }, 80); // ~12.5fps simulated
  };

  const stopPipeline = () => {
    setRunning(false);
    clearInterval(intervalRef.current);
  };

  useEffect(() => () => clearInterval(intervalRef.current), []);

  // Unique MTM codes seen
  const uniqueCodes = [...new Set(mtmLog.map(e => e.mtm_code))];

  return (
    <div style={{
      background: COLORS.bg, minHeight: "100vh", color: COLORS.text,
      fontFamily: "'Courier New', monospace", padding: 20,
    }}>
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateX(-6px); } to { opacity: 1; } }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        ::-webkit-scrollbar { width: 4px; } 
        ::-webkit-scrollbar-track { background: ${COLORS.bg}; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 2px; }
      `}</style>

      {/* ── HEADER ── */}
      <div style={{ marginBottom: 20, borderBottom: `1px solid ${COLORS.border}`, paddingBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
          <div>
            <div style={{ fontSize: 9, letterSpacing: 4, color: COLORS.accent, marginBottom: 4 }}>
              INDUSTRIAL MOTION ANALYSIS SYSTEM v1.0 — PHASE 1
            </div>
            <h1 style={{ margin: 0, fontSize: 20, fontWeight: 900, letterSpacing: 1, color: COLORS.text }}>
              YOLOv8-POSE → SKELETON SEQUENCE BUILDER
            </h1>
            <div style={{ fontSize: 10, color: COLORS.textDim, marginTop: 4 }}>
              MTM Code Extraction Pipeline · ST-GCN Ready Output
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={running ? stopPipeline : startPipeline}
              style={{
                padding: "10px 24px", borderRadius: 4, border: "none", cursor: "pointer",
                background: running ? COLORS.red : COLORS.accent,
                color: "#000", fontWeight: 800, fontFamily: "monospace",
                letterSpacing: 2, fontSize: 11,
                boxShadow: running ? `0 0 20px ${COLORS.red}44` : `0 0 20px ${COLORS.accent}44`,
                transition: "all 0.2s ease"
              }}
            >
              {running ? "■ STOP" : "▶ START PIPELINE"}
            </button>
          </div>
        </div>
      </div>

      {/* ── PIPELINE STAGES ── */}
      <div style={{ marginBottom: 16 }}>
        <PipelineStages activeStage={running ? activeStage : -1} />
      </div>

      {/* ── STATS ROW ── */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        <StatCard label="FRAME" value={String(frameIdx).padStart(4, "0")} sub={`/ ${totalFrames}`} />
        <StatCard label="PROC FPS" value={running ? fps : "—"} sub="frames/sec" color={COLORS.green} />
        <StatCard label="PERSONS" value={currentFrame?.detected_persons ?? 0} sub="detected" color={COLORS.blue} />
        <StatCard label="MTM CODES" value={uniqueCodes.length} sub="unique found" color={COLORS.accent} />
        <StatCard
          label="CONFIDENCE"
          value={currentFrame ? `${(currentFrame.confidence * 100).toFixed(0)}%` : "—"}
          sub="action score"
          color={currentFrame?.confidence > 0.85 ? COLORS.green : COLORS.accent}
        />
      </div>

      {/* ── MAIN GRID ── */}
      <div style={{ display: "grid", gridTemplateColumns: "200px 1fr 1fr", gap: 12, alignItems: "start" }}>

        {/* COL 1: Skeleton Viz */}
        <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 6, padding: 12 }}>
          <div style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: 2, marginBottom: 8 }}>SKELETON VIEW</div>
          <SkeletonCanvas frame={currentFrame} width={176} height={260} />
          {currentFrame && (
            <div style={{ marginTop: 8 }}>
              <ConfBar value={currentFrame.confidence} label="ACTION CONF" />
              <ConfBar value={currentFrame.keypoints[9].conf} label="L.WRIST" color={COLORS.green} />
              <ConfBar value={currentFrame.keypoints[10].conf} label="R.WRIST" color={COLORS.green} />
              <ConfBar value={currentFrame.keypoints[15].conf} label="L.ANKLE" color={COLORS.blue} />
            </div>
          )}
        </div>

        {/* COL 2: Keypoint Data */}
        <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 6, padding: 12 }}>
          <div style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: 2, marginBottom: 8 }}>
            KEYPOINT TENSOR — FRAME {String(frameIdx).padStart(4, "0")} @ {currentFrame?.timestamp}s
          </div>

          {/* Current Action Banner */}
          {currentFrame && (
            <div style={{
              background: COLORS.accent + "18",
              border: `1px solid ${COLORS.accent}44`,
              borderRadius: 4, padding: "8px 12px", marginBottom: 12,
              animation: "fadeIn 0.2s ease"
            }}>
              <div style={{ fontSize: 8, color: COLORS.textDim, letterSpacing: 2 }}>CURRENT MTM CODE</div>
              <div style={{ fontSize: 16, fontWeight: 900, color: COLORS.accent, letterSpacing: 1 }}>
                {currentFrame.mtm_code}
              </div>
              <div style={{ fontSize: 8, color: COLORS.muted }}>
                raw: {currentFrame.action_raw}
              </div>
            </div>
          )}

          <KeypointTable frame={currentFrame} />

          {/* Skeleton JSON preview */}
          {currentFrame && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: 2, marginBottom: 6 }}>
                ST-GCN INPUT VECTOR (sample)
              </div>
              <div style={{
                background: COLORS.bg, borderRadius: 4, padding: 8,
                fontSize: 9, color: COLORS.green, overflowX: "auto",
                border: `1px solid ${COLORS.border}`
              }}>
                <pre style={{ margin: 0 }}>{JSON.stringify({
                  frame: frameIdx,
                  t: parseFloat(currentFrame.timestamp),
                  nodes: currentFrame.keypoints.slice(0, 5).map((k, i) => ({
                    id: i, x: +k.x.toFixed(3), y: +k.y.toFixed(3), c: +k.conf.toFixed(2)
                  }))
                }, null, 2)}</pre>
              </div>
            </div>
          )}
        </div>

        {/* COL 3: MTM Log + Code Legend */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 6, padding: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: 2 }}>MTM SEQUENCE LOG</div>
              {running && (
                <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <div style={{
                    width: 6, height: 6, borderRadius: "50%", background: COLORS.green,
                    animation: "pulse 1s infinite"
                  }} />
                  <span style={{ fontSize: 8, color: COLORS.green }}>LIVE</span>
                </div>
              )}
            </div>
            <MTMLog entries={mtmLog} />
          </div>

          {/* Detected codes summary */}
          <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 6, padding: 12 }}>
            <div style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: 2, marginBottom: 8 }}>
              CODES DETECTED THIS SESSION
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {uniqueCodes.length === 0 ? (
                <div style={{ color: COLORS.muted, fontSize: 10 }}>— start pipeline to detect codes —</div>
              ) : uniqueCodes.map((code, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "4px 6px", borderRadius: 3,
                  background: COLORS.bg, fontSize: 10,
                  border: `1px solid ${COLORS.border}`,
                }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: code.includes("WALK") ? COLORS.blue :
                                code.includes("GRASP") ? COLORS.green :
                                code.includes("HOLD") ? COLORS.accent : COLORS.textDim
                  }} />
                  <span style={{ color: COLORS.text }}>{code}</span>
                  <span style={{ marginLeft: "auto", color: COLORS.textDim, fontSize: 9 }}>
                    {mtmLog.filter(e => e.mtm_code === code).length}×
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── FOOTER ── */}
      <div style={{
        marginTop: 20, paddingTop: 12, borderTop: `1px solid ${COLORS.border}`,
        display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8
      }}>
        <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>
          PHASE 1 OF 5 · NEXT: ST-GCN TRAINING → MS-TCN SMOOTHING → CLAUDE API MTM FORMATTER
        </div>
        <div style={{ fontSize: 9, color: COLORS.muted }}>
          STACK: YOLOv8-POSE · BYTETRACK · SKELETON-SEQ-BUILDER · COCO-17KP
        </div>
      </div>
    </div>
  );
}
