export interface CreateSessionResponse {
  session_id: string;
  model_name: string;
}

import type { PackedMask } from "./mask";

export interface PredictResponse {
  masks: PackedMask[];
  objectMasks?: Record<number, PackedMask>;
  objectHistory?: Record<string, Partial<ObjectHistoryEntry>>;
}

export interface ObjectHistoryEntry {
  undo: number;
  redo: number;
  strokes: number;
  has_mask: boolean;
}

export interface Sam3PromptResponse {
  masks: PackedMask[];
  scores: number[];
  bboxes: number[][];
}

export interface UploadResponse {
  width: number;
  height: number;
}

export interface StrokePoint {
  x: number;
  y: number;
}

export interface CanvasState {
  imageWidth: number;
  imageHeight: number;
  canvasWidth: number;
  canvasHeight: number;
  scale: number;
  offsetX: number;
  offsetY: number;
}

export interface DetectorInfo {
  name: string;
  display_name: string;
  type: "grounding" | "yoloe";
  loaded: boolean;
  downloaded: boolean;
  tier?: string;
  perf?: string;
}

export interface Detection {
  bbox: [number, number, number, number];
  score: number;
  label: string;
  mask?: string | null;
}

export interface DetectResponse {
  detections: Detection[];
}
