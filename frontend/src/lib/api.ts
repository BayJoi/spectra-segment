import type {
  CreateSessionResponse,
  DetectResponse,
  DetectorInfo,
  PredictResponse,
  Sam3PromptResponse,
  UploadResponse,
} from "./types";
import type { PackedMask } from "./mask";

export const BASE = "http://localhost:8000";

interface RawPredictResponse {
  masks: string[];
  object_masks?: Record<string, string>;
}

interface RawSam3PromptResponse {
  masks: string[];
  scores: number[];
  bboxes: number[][];
}

async function decodeMasks(masks: string[]): Promise<PackedMask[]> {
  return Promise.all(masks.map((m) => decodeDetectionMask(m)));
}

async function decodeObjectMasks(obj?: Record<string, string>): Promise<Record<number, PackedMask>> {
  if (!obj) return {};
  const out: Record<number, PackedMask> = {};
  for (const [k, v] of Object.entries(obj)) {
    out[Number(k)] = await decodeDetectionMask(v);
  }
  return out;
}

async function toPredictResponse(r: RawPredictResponse): Promise<PredictResponse> {
  return {
    masks: await decodeMasks(r.masks),
    objectMasks: await decodeObjectMasks(r.object_masks),
  };
}

let _token: string | null = null;

async function getToken(): Promise<string> {
  if (_token) return _token;
  const res = await fetch(`${BASE}/api/token`);
  if (!res.ok) throw new Error("Failed to get local token");
  const data = await res.json();
  _token = data.token;
  return _token!;
}

async function apiFetch<T>(path: string, init?: RequestInit, asBlob?: boolean): Promise<T> {
  const doFetch = async (token: string) => {
    const headers = new Headers(init?.headers);
    headers.set("X-Local-Token", token);
    return fetch(`${BASE}${path}`, { ...init, headers });
  };

  let res = await doFetch(await getToken());
  if (res.status === 401) {
    _token = null;
    res = await doFetch(await getToken());
  }

  if (!res.ok) {
    const body = (await res.text()).replace(/[\x00-\x1f\x7f]/g, " ").slice(0, 500).trim();
    throw new Error(`API ${res.status}: ${body || "(no response body)"}`);
  }
  return (asBlob ? res.blob() : res.json()) as Promise<T>;
}

export const api = {
  createSession: (modelName: string) =>
    apiFetch<CreateSessionResponse>("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName }),
    }),

  destroySession: (sessionId: string) =>
    apiFetch<{ status: string }>(`/api/sessions/${sessionId}`, {
      method: "DELETE",
    }),

  releaseSession: (sessionId: string): void => {
    const base = `${BASE}/api/sessions/${sessionId}`;
    if (typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
      try {
        navigator.sendBeacon(`${base}/release`, "release");
      } catch {}
    }
    const token = _token;
    if (token) {
      fetch(`${base}`, {
        method: "DELETE",
        keepalive: true,
        headers: { "X-Local-Token": token },
      }).catch(() => {});
    }
  },

  sessionHealth: async (sessionId: string): Promise<boolean> => {
    try {
      await apiFetch<{ session_id: string }>(`/api/sessions/${sessionId}`);
      return true;
    } catch {
      return false;
    }
  },

  uploadImage: async (sessionId: string, file: File): Promise<UploadResponse> => {
    const upload = async (token: string) => {
      const fd = new FormData();
      fd.append("file", file);
      return fetch(`${BASE}/api/sessions/${sessionId}/image`, {
        method: "POST",
        body: fd,
        headers: { "X-Local-Token": token },
      });
    };

    let res = await upload(await getToken());
    if (res.status === 401) {
      _token = null;
      res = await upload(await getToken());
    }

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Upload failed: ${res.status} ${text}`);
    }
    return res.json();
  },

  predict: (
    sessionId: string,
    params: {
      object_id?: number;
      points?: number[][];
      labels?: number[];
      bboxes?: number[];
    }
  ) =>
    apiFetch<RawPredictResponse>(`/api/sessions/${sessionId}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }).then(toPredictResponse),

  undo: (sessionId: string, objectId: number = 0) =>
    apiFetch<RawPredictResponse>(
      `/api/sessions/${sessionId}/undo?object_id=${objectId}`,
      { method: "POST" }
    ).then(toPredictResponse),

  redo: (sessionId: string, objectId: number = 0) =>
    apiFetch<RawPredictResponse>(
      `/api/sessions/${sessionId}/redo?object_id=${objectId}`,
      { method: "POST" }
    ).then(toPredictResponse),

  clearObject: (sessionId: string, objectId: number = 0) =>
    apiFetch<RawPredictResponse>(
      `/api/sessions/${sessionId}/clear-object?object_id=${objectId}`,
      { method: "POST" }
    ).then(toPredictResponse),

  sam3Prompt: (sessionId: string, text: string) =>
    apiFetch<RawSam3PromptResponse>(`/api/sessions/${sessionId}/sam3-prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).then(async (r): Promise<Sam3PromptResponse> => ({ masks: await decodeMasks(r.masks), scores: r.scores, bboxes: r.bboxes })),

  sam3Undo: (sessionId: string) =>
    apiFetch<RawSam3PromptResponse>(`/api/sessions/${sessionId}/sam3-undo`, {
      method: "POST",
    }).then(async (r): Promise<Sam3PromptResponse> => ({ masks: await decodeMasks(r.masks), scores: r.scores, bboxes: r.bboxes })),

  sam3Redo: (sessionId: string) =>
    apiFetch<RawSam3PromptResponse>(`/api/sessions/${sessionId}/sam3-redo`, {
      method: "POST",
    }).then(async (r): Promise<Sam3PromptResponse> => ({ masks: await decodeMasks(r.masks), scores: r.scores, bboxes: r.bboxes })),

  sam3RemoveInstance: (sessionId: string, promptIndex: number, instanceIndex: number) =>
    apiFetch<RawSam3PromptResponse>(`/api/sessions/${sessionId}/sam3-remove-instance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt_index: promptIndex, instance_index: instanceIndex }),
    }).then(async (r): Promise<Sam3PromptResponse> => ({ masks: await decodeMasks(r.masks), scores: r.scores, bboxes: r.bboxes })),

  setSam3Settings: (params: { keep_loaded?: boolean; encode_dim?: number }) =>
    apiFetch<{ keep_loaded: boolean; encode_dim: number }>("/api/settings/sam3", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),

  listDetectors: () => apiFetch<DetectorInfo[]>("/api/detectors"),

  loadDetector: (detectorName: string) =>
    apiFetch<{ status: string; detector: string }>("/api/detectors/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ detector_name: detectorName }),
    }),

  detect: (
    sessionId: string,
    params: { query: string; max_detections?: number; use_yoloe_masks?: boolean }
  ) =>
    apiFetch<DetectResponse>(`/api/sessions/${sessionId}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),

  exportZip: (
    sessionId: string,
    params: {
      root: string;
      files: { path: string; mask_b64: string }[];
      include_whole?: boolean;
      format?: string;
      background_color?: number[];
      feather_radius?: number;
    }
  ) =>
    apiFetch<{ data: string; format: string; count: number }>(
      `/api/sessions/${sessionId}/export-zip`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      }
    ),

  getLogs: (limit: number = 300) =>
    apiFetch<{ entries: { level: string; message: string }[]; total: number }>(
      `/api/logs?limit=${limit}`
    ),

  getModels: () =>
    apiFetch<{ models: { name: string; display_name: string; type: string; downloaded: boolean; loaded: boolean; tier: string; perf: string }[] }>(
      "/api/models/status"
    ),

  segmentBatch: (sessionId: string, bboxes: number[][]) =>
    apiFetch<RawPredictResponse>(`/api/sessions/${sessionId}/segment-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bboxes }),
    }).then(async (r) => ({ masks: await decodeMasks(r.masks) })),
};

export async function decodeDetectionMask(base64Png: string): Promise<PackedMask> {
  const binary = atob(base64Png);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const blob = new Blob([bytes], { type: "image/png" });
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = reject;
      img.src = url;
    });
    if (img.width > 8192 || img.height > 8192 || img.width * img.height > 8192 * 8192) {
      throw new Error("Mask dimensions too large");
    }
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const data = imageData.data;
    const width = img.width;
    const height = img.height;
    const packed = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      const off = y * width;
      const rowOffset = y * width * 4;
      for (let x = 0; x < width; x++) {
        if (data[rowOffset + x * 4] > 127) packed[off + x] = 1;
      }
    }
    return { w: width, h: height, data: packed };
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function encodeMaskPng(mask: PackedMask): string {
  const h = mask.h;
  const w = mask.w;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;
  const src = mask.data;
  for (let y = 0; y < h; y++) {
    const rowOffset = y * w;
    const rowOffset4 = rowOffset * 4;
    for (let x = 0; x < w; x++) {
      if (src[rowOffset + x]) {
        const idx = rowOffset4 + x * 4;
        data[idx] = 255;
        data[idx + 1] = 255;
        data[idx + 2] = 255;
        data[idx + 3] = 255;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const dataUrl = canvas.toDataURL("image/png");
  return dataUrl.slice(dataUrl.indexOf(",") + 1);
}
