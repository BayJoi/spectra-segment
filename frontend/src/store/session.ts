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
