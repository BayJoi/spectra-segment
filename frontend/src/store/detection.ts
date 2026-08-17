import { atom } from "jotai";
import type { Detection, DetectorInfo } from "@/lib/types";

export const detectModeAtom = atom(false);
export const toolModeAtom = atom<"positive" | "negative">("positive");
export const brushSizeAtom = atom(20);
export const featherRadiusAtom = atom(0);

export const detectorsAtom = atom<DetectorInfo[]>([]);
export const selectedDetectorAtom = atom("");
export const loadedDetectorAtom = atom<string | null>(null);
export const detectorLoadingAtom = atom(false);

export const detectQueryAtom = atom("");
export const detectionsAtom = atom<Detection[]>([]);
export const selectedDetectionAtom = atom<number | null>(null);
export const isDetectingAtom = atom(false);

export const yoloeMasksEnabledAtom = atom(false);
