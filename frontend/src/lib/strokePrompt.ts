import type { StrokePoint } from "./types";

type StrokeMode = "positive" | "negative";

interface StrokePromptResult {
  points: number[][];
  labels: number[];
}

const MAX_RING_POINTS = 32;
const MAX_INTERIOR_POINTS = 8;
const MAX_AREA_POINTS = 12;
const AREA_ERODE_FRACTION = 0.3;
const RASTER_MAX_DIM = 1024;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function dist(a: StrokePoint, b: StrokePoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function roundPoint(p: StrokePoint, imgW: number, imgH: number): [number, number] {
  return [
    Math.round(Math.max(0, Math.min(imgW - 1, p.x))),
    Math.round(Math.max(0, Math.min(imgH - 1, p.y))),
  ];
}

function downsample(points: StrokePoint[], spacing: number): StrokePoint[] {
  const out: StrokePoint[] = [];
  for (const p of points) {
    const prev = out[out.length - 1];
    if (!prev || dist(prev, p) >= spacing) out.push(p);
  }
  return out;
}

function capPoints(points: number[][], max: number): number[][] {
  if (points.length <= max) return points;
  const stride = Math.ceil(points.length / max);
  const down: number[][] = [];
  for (let i = 0; i < points.length; i += stride) down.push(points[i]);
  return down;
}

function buildRingPrompt(
  stroke: StrokePoint[],
  brushSize: number,
  imgW: number,
  imgH: number,
): StrokePromptResult {
  const minDim = Math.min(imgW, imgH);
  const spacing = Math.max(6, Math.round(minDim / 96));
  const ring = capPoints(
    downsample(stroke, spacing).map((p) => roundPoint(p, imgW, imgH)),
    MAX_RING_POINTS,
  );

  if (ring.length < 3) {
    return { points: [ring[0] ?? roundPoint(stroke[0], imgW, imgH)], labels: [1] };
  }

  let cx = 0;
  let cy = 0;
  for (const [x, y] of ring) {
    cx += x;
    cy += y;
  }
  cx /= ring.length;
  cy /= ring.length;

  let avgRadius = 0;
  for (const [x, y] of ring) avgRadius += Math.hypot(x - cx, y - cy);
  avgRadius = Math.max(1, avgRadius / ring.length);

  const shrink = clamp(brushSize / avgRadius, 0.15, 0.5);

  const interior: number[][] = [[Math.round(cx), Math.round(cy)]];
  const step = Math.max(1, Math.ceil(ring.length / MAX_INTERIOR_POINTS));
  for (let i = 0; i < ring.length; i += step) {
    const [x, y] = ring[i];
    interior.push([
      Math.round(cx + (x - cx) * (1 - shrink)),
      Math.round(cy + (y - cy) * (1 - shrink)),
    ]);
  }

  const positives = capPoints(interior, MAX_INTERIOR_POINTS);
  return {
    points: [...positives, ...ring],
    labels: [...positives.map(() => 1), ...ring.map(() => 0)],
  };
}

function buildAreaPrompt(
  stroke: StrokePoint[],
  brushSize: number,
  mode: StrokeMode,
  imgW: number,
  imgH: number,
): StrokePromptResult {
  const label = mode === "positive" ? 1 : 0;

  const maxDim = Math.max(imgW, imgH, 1);
  const scale = RASTER_MAX_DIM / maxDim;
  const rw = Math.max(1, Math.round(imgW * scale));
  const rh = Math.max(1, Math.round(imgH * scale));

  const canvas = document.createElement("canvas");
  canvas.width = rw;
  canvas.height = rh;
  const ctx = canvas.getContext("2d");
  if (!ctx) return { points: [roundPoint(stroke[0], imgW, imgH)], labels: [label] };

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, rw, rh);
  ctx.strokeStyle = "#fff";
  const erode = brushSize * AREA_ERODE_FRACTION;
  ctx.lineWidth = Math.max(1, brushSize * scale - 2 * erode * scale);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  stroke.forEach((p, i) => {
    const px = p.x * scale;
    const py = p.y * scale;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();

  const image = ctx.getImageData(0, 0, rw, rh);
  const data = image.data;
  const step = Math.max(4, Math.round(16 * scale));
  const collected: number[][] = [];
  for (let y = 0; y < rh; y += step) {
    for (let x = 0; x < rw; x += step) {
      if (data[(y * rw + x) * 4] > 128) {
        collected.push([Math.round(x / scale), Math.round(y / scale)]);
      }
    }
  }

  const pts = capPoints(collected, MAX_AREA_POINTS);
  if (pts.length === 0) {
    return { points: [roundPoint(stroke[0], imgW, imgH)], labels: [label] };
  }
  return { points: pts, labels: pts.map(() => label) };
}

export function buildStrokePrompt(
  stroke: StrokePoint[],
  brushSize: number,
  mode: StrokeMode,
  imgW: number,
  imgH: number,
): StrokePromptResult {
  if (stroke.length === 0) return { points: [], labels: [] };
  if (stroke.length < 3) {
    const label = mode === "positive" ? 1 : 0;
    return { points: [roundPoint(stroke[0], imgW, imgH)], labels: [label] };
  }
  if (mode === "negative") {
    return buildAreaPrompt(stroke, brushSize, "negative", imgW, imgH);
  }

  const first = stroke[0];
  const last = stroke[stroke.length - 1];
  const gap = dist(first, last);
  const minX = Math.min(...stroke.map((p) => p.x));
  const maxX = Math.max(...stroke.map((p) => p.x));
  const minY = Math.min(...stroke.map((p) => p.y));
  const maxY = Math.max(...stroke.map((p) => p.y));
  const bboxDiag = Math.hypot(maxX - minX, maxY - minY);
  const closed = gap <= Math.max(brushSize * 1.5, 4) || gap <= 0.12 * bboxDiag;

  if (closed && stroke.length >= 4) {
    return buildRingPrompt(stroke, brushSize, imgW, imgH);
  }
  return buildAreaPrompt(stroke, brushSize, "positive", imgW, imgH);
}
