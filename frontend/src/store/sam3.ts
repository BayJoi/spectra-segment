import { atom } from "jotai";
import type { PackedMask } from "@/lib/mask";

export interface Sam3Prompt {
  text: string;
  masks: PackedMask[];
  scores: number[];
  bboxes: number[][];
}

export interface Sam3Instance {
  promptIndex: number;
  instanceIndex: number;
  text: string;
  bbox: number[];
  score: number;
}

export const sam3ModeAtom = atom(false);

export const sam3PromptsAtom = atom<Sam3Prompt[]>([]);

export const sam3RedoStackAtom = atom<Sam3Prompt[]>([]);

export const sam3PromptInputAtom = atom("");

export const sam3PromptingAtom = atom(false);

export const sam3InstancesAtom = atom<Sam3Instance[]>([]);

export const selectedSam3InstanceAtom = atom<{ promptIndex: number; instanceIndex: number } | null>(null);

export type Sam3EncodeDim = 512 | 1024 | 1500;

export const sam3KeepLoadedAtom = atom(true);

export const sam3EncodeDimAtom = atom<Sam3EncodeDim>(1024);
