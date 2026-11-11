import { loadImageElement } from "@/lib/fileExport";

export type MatteGuide = "auto" | "stroke" | "text" | "box";

export interface MatteModelInfo {
  name: string;
  display_name: string;
  input_size: number;
  tier: string;
  perf: string;
  soft_alpha: boolean;
  weight_mb: number;
  downloaded: boolean;
  loaded: boolean;
}

export interface MatteResponse {
  alpha: string;
  width: number;
  height: number;
  model: string;
  guide: MatteGuide;
  box: number[] | null;
  ms: number;
}

export interface MatteAlpha {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  revision: number;
}

export interface MatteStrokeSet {
  points: number[][];
  labels: number[];
}

export interface MatteInfo {
  model: string;
  guide: MatteGuide;
  ms: number;
}

export interface MatteSnapshot {
  alphaB64: string | null;
  strokes: MatteStrokeSet;
  guide: MatteGuide;
  info: MatteInfo | null;
}

export interface MatteLayer {
  id: number;
  name: string;
  color: string;
  model: string;
  visible: boolean;
  rawAlpha: MatteAlpha | null;
  alphaB64: string | null;
  strokes: MatteStrokeSet;
  guide: MatteGuide;
  info: MatteInfo | null;
  history: MatteSnapshot[];
  redo: MatteSnapshot[];
}

export const MATTE_LAYER_COLORS = [
  "#d47534",
  "#5aa9a3",
  "#c9a35c",
  "#8a8fd4",
  "#c47d8f",
  "#7fb069",
] as const;

export function matteLayerName(id: number): string {
  return id === 0 ? "Main subject" : `Subject ${id + 1}`;
}

export function matteLayerColor(id: number): string {
  return MATTE_LAYER_COLORS[id % MATTE_LAYER_COLORS.length];
}

export function emptyMatteStrokes(): MatteStrokeSet {
  return { points: [], labels: [] };
}

export function createMatteLayer(id: number, model = "birefnet-general"): MatteLayer {
  return {
    id,
    name: matteLayerName(id),
    color: matteLayerColor(id),
    model,
    visible: true,
    rawAlpha: null,
    alphaB64: null,
    strokes: emptyMatteStrokes(),
    guide: "auto",
    info: null,
    history: [],
    redo: [],
  };
}

export async function decodeAlphaPng(base64Png: string, revision: number): Promise<MatteAlpha> {
  const img = await loadImageElement(`data:image/png;base64,${base64Png}`);
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas unavailable");

  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const px = imageData.data;
  for (let i = 0; i < px.length; i += 4) {
    const value = px[i];
    px[i] = 255;
    px[i + 1] = 255;
    px[i + 2] = 255;
    px[i + 3] = value;
  }
  ctx.putImageData(imageData, 0, 0);
  return { canvas, width: canvas.width, height: canvas.height, revision };
}

const shiftCache = new WeakMap<HTMLCanvasElement, Map<string, HTMLCanvasElement>>();
const MAX_SHIFT_CACHE = 4;

function boxBlurAlpha(source: Float32Array, width: number, height: number, radius: number): Float32Array {
  const tmp = new Float32Array(source.length);
  const out = new Float32Array(source.length);
  const window = radius * 2 + 1;

  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;
    for (let x = -radius; x <= radius; x++) if (x >= 0 && x < width) sum += source[row + x];
    for (let x = 0; x < width; x++) {
      tmp[row + x] = sum / window;
      const add = x + radius + 1;
      const remove = x - radius;
      if (add < width) sum += source[row + add];
      if (remove >= 0) sum -= source[row + remove];
    }
  }

  for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let y = -radius; y <= radius; y++) if (y >= 0 && y < height) sum += tmp[y * width + x];
    for (let y = 0; y < height; y++) {
      out[y * width + x] = sum / window;
      const add = y + radius + 1;
      const remove = y - radius;
      if (add < height) sum += tmp[add * width + x];
      if (remove >= 0) sum -= tmp[remove * width + x];
    }
  }
  return out;
}

export function mergeAlphaCanvases(canvases: readonly HTMLCanvasElement[]): HTMLCanvasElement | null {
  const usable = canvases.filter((canvas) => canvas.width > 0 && canvas.height > 0);
  if (usable.length === 0) return null;
  if (usable.length === 1) return usable[0];
  const width = Math.max(...usable.map((canvas) => canvas.width));
  const height = Math.max(...usable.map((canvas) => canvas.height));
  const merged = document.createElement("canvas");
  merged.width = width;
  merged.height = height;
  const ctx = merged.getContext("2d");
  if (!ctx) return null;
  for (const canvas of usable) ctx.drawImage(canvas, 0, 0, width, height);
  return merged;
}

function shiftedAlpha(source: HTMLCanvasElement, edgeShift: number, invert: boolean): HTMLCanvasElement {
  if (edgeShift === 0 && !invert) return source;
  const key = `${edgeShift}:${invert ? 1 : 0}`;
  let cache = shiftCache.get(source);
  if (!cache) {
    cache = new Map();
    shiftCache.set(source, cache);
  }
  const cached = cache.get(key);
  if (cached) return cached;

  const base = document.createElement("canvas");
  base.width = source.width;
  base.height = source.height;
  const ctx = base.getContext("2d");
  if (!ctx || source.width === 0 || source.height === 0) return base;
  ctx.drawImage(source, 0, 0);

  const imageData = ctx.getImageData(0, 0, source.width, source.height);
  const px = imageData.data;
  const shift = edgeShift / 100;
  const lut = new Uint8ClampedArray(256);
  for (let value = 0; value < 256; value++) {
    let next = value / 255 + shift;
    next = next < 0 ? 0 : next > 1 ? 1 : next;
    if (invert) next = 1 - next;
    lut[value] = Math.round(next * 255);
  }
  for (let i = 3; i < px.length; i += 4) px[i] = lut[px[i]];
  ctx.putImageData(imageData, 0, 0);

  cache.set(key, base);
  if (cache.size > MAX_SHIFT_CACHE) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
  return base;
}

export function applyAlphaControls(
  source: HTMLCanvasElement,
  edgeShift: number,
  edgeFeather: number,
  invert: boolean
): HTMLCanvasElement {
  const width = source.width;
  const height = source.height;
  if (width === 0 || height === 0) return source;
  const base = shiftedAlpha(source, edgeShift, invert);

  if (edgeFeather <= 0) return base;

  const out = document.createElement("canvas");
  out.width = width;
  out.height = height;
  const ctx = out.getContext("2d");
  if (!ctx) return base;

  if (typeof ctx.filter === "string") {
    ctx.filter = `blur(${Math.max(0.5, edgeFeather / 2)}px)`;
    ctx.drawImage(base, 0, 0);
    ctx.filter = "none";
    return out;
  }

  ctx.drawImage(base, 0, 0);
  const imageData = ctx.getImageData(0, 0, width, height);
  const px = imageData.data;
  let alpha: Float32Array<ArrayBufferLike> = new Float32Array(width * height);
  for (let i = 0, p = 3; i < alpha.length; i++, p += 4) alpha[i] = px[p] / 255;
  const radius = Math.max(1, Math.round(edgeFeather / 2));
  for (let pass = 0; pass < 3; pass++) alpha = boxBlurAlpha(alpha, width, height, radius);
  for (let i = 0, p = 3; i < alpha.length; i++, p += 4) px[p] = Math.round(alpha[i] * 255);
  ctx.putImageData(imageData, 0, 0);
  return out;
}

