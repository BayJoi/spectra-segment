import { atom } from "jotai";
import type { PackedMask } from "@/lib/mask";

export const sessionIdAtom = atom<string | null>(null);
export const modelNameAtom = atom("");

export const imageUrlAtom = atom<string | null>(null);
export const imageWidthAtom = atom(0);
export const imageHeightAtom = atom(0);
export const imageFileAtom = atom<File | null>(null);

export const hasImageAtom = atom((get) => !!get(imageUrlAtom));

export const masksAtom = atom<PackedMask[]>([]);

export const objectMasksAtom = atom<Record<number, PackedMask>>({});

export const perDetectionMasksAtom = atom<Record<number, PackedMask>>({});

export const brushObjectsAtom = atom<number[]>([0]);
export const activeObjectIdAtom = atom<number>(0);
export const objectUndoCountsAtom = atom<Record<number, number>>({});
export const objectRedoCountsAtom = atom<Record<number, number>>({});

export interface ObjectHistory {
  undo: number;
  redo: number;
  strokes: number;
  has_mask: boolean;
}

export type ObjectHistoryEntry = Partial<ObjectHistory>;

export const objectHistoryAtom = atom<Record<number, ObjectHistory>>({});
interface SubjectMeta {
  id: number;
  name: string;
  color: string;
}

const SUBJECT_COLORS = [
  "#866647",
  "#aea447",
  "#7aae47",
  "#47ae51",
  "#47ae8f",
  "#478fae",
  "#4751ae",
  "#7a47ae",
  "#ae47a4",
  "#ae4766",
] as const;

export const subjectMetaAtom = atom<Record<number, SubjectMeta>>({});

function shiftLightness(hex: string, amount: number): string {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const light = (max + min) / 2;
  const span = max - min;
  const sat = span === 0 ? 0 : span / (1 - Math.abs(2 * light - 1));
  let hue = 0;
  if (span !== 0) {
    if (max === r) hue = ((g - b) / span) % 6;
    else if (max === g) hue = (b - r) / span + 2;
    else hue = (r - g) / span + 4;
    hue *= 60;
    if (hue < 0) hue += 360;
  }
  const target = Math.min(0.85, Math.max(0.15, light + amount));
  const chroma = (1 - Math.abs(2 * target - 1)) * sat;
  const second = chroma * (1 - Math.abs(((hue / 60) % 2) - 1));
  const offset = target - chroma / 2;
  const [rr, gg, bb] =
    hue < 60 ? [chroma, second, 0]
    : hue < 120 ? [second, chroma, 0]
    : hue < 180 ? [0, chroma, second]
    : hue < 240 ? [0, second, chroma]
    : hue < 300 ? [second, 0, chroma]
    : [chroma, 0, second];
  const channel = (v: number) =>
    Math.round(Math.min(1, Math.max(0, v + offset)) * 255).toString(16).padStart(2, "0");
  return `#${channel(rr)}${channel(gg)}${channel(bb)}`;
}

const subjectColorCache = new Map<number, string>();

export function subjectColor(id: number): string {
  const cached = subjectColorCache.get(id);
  if (cached !== undefined) return cached;
  const base = SUBJECT_COLORS[id % SUBJECT_COLORS.length];
  const cycle = Math.floor(id / SUBJECT_COLORS.length);
  const color =
    cycle === 0
      ? base
      : shiftLightness(base, (cycle % 2 === 1 ? -1 : 1) * 0.13 * Math.ceil(cycle / 2));
  subjectColorCache.set(id, color);
  return color;
}

export function subjectName(id: number): string {
  return `Subject ${id + 1}`;
}

export const sam3ReadyAtom = atom(false);

export interface ModelInfo {
  name: string;
  display_name: string;
  type: string;
  detector_type?: string;
  downloaded: boolean;
  loaded: boolean;
  tier: string;
  perf: string;
}

export const modelsAtom = atom<ModelInfo[]>([]);
