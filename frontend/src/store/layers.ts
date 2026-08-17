import { atom } from "jotai";

export interface Layer {
  id: string;
  type: "brush" | "detection" | "sam3";
  label: string;
  preview: string | null;
  objectId: number;
  detectionIndex?: number;
  promptIndex?: number;
  instanceIndex?: number;
  createdAt: number;
}

export const layersAtom = atom<Layer[]>([]);
export const selectedLayersAtom = atom<Set<string>>(new Set<string>());
export const layerIdCounterAtom = atom(0);
