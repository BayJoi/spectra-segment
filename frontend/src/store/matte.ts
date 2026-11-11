import { atom } from "jotai";
import { atomWithStorage } from "jotai/utils";
import {
  createMatteLayer,
  type MatteGuide,
  type MatteLayer,
  type MatteModelInfo,
} from "@/lib/matte";
import { modeLockAtom } from "./ui";

export const matteModeAtom = atom((get) => get(modeLockAtom) === "matte");

export const matteModelsAtom = atom<MatteModelInfo[]>([]);
export const matteModelAtom = atomWithStorage("matteModel", "birefnet-general");

export const matteLayersAtom = atom<MatteLayer[]>([createMatteLayer(0)]);
export const matteActiveLayerIdAtom = atom(0);

export const matteGuideAtom = atom<MatteGuide>("auto");

export const matteStrokesAtom = atom<{ points: number[][]; labels: number[] }>({
  points: [],
  labels: [],
});

export const matteBusyAtom = atom(false);
export const matteHistoryBusyAtom = atom(false);
export const matteInfoAtom = atom<{ model: string; guide: MatteGuide; ms: number } | null>(null);

export const matteEdgeShiftAtom = atom(0);
export const matteEdgeFeatherAtom = atom(0);
export const matteInvertAtom = atom(false);

export const matteExportOpenAtom = atom(false);
