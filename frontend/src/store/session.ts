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
export const brushPredictInFlightAtom = atom(0);

export interface ObjectHistory {
  undo: number;
  redo: number;
  strokes: number;
  has_mask: boolean;
}

export type ObjectHistoryEntry = Partial<ObjectHistory>;

export const objectHistoryAtom = atom<Record<number, ObjectHistory>>({});
export interface SubjectMeta {
  id: number;
  name: string;
  color: string;
}

export const SUBJECT_COLORS = [
  "#b85c2a",
  "#6b8fb5",
  "#6f9a6b",
  "#9a7fb5",
  "#b5809a",
  "#b5a36b",
  "#5f9a96",
  "#a8705f",
] as const;

export const subjectMetaAtom = atom<Record<number, SubjectMeta>>({});

export function subjectColor(id: number): string {
  return SUBJECT_COLORS[id % SUBJECT_COLORS.length];
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
