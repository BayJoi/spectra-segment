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

let _layerSeq = 0;

export function nextLayerId(existing: Layer[] = []): string {
  let maxId = 0;
  for (const l of existing) {
    const n = Number(l.id.replace(/^layer-/, ""));
    if (Number.isFinite(n) && n > maxId) maxId = n;
  }
  if (maxId >= _layerSeq) _layerSeq = maxId + 1;
  return `layer-${_layerSeq++}`;
}
