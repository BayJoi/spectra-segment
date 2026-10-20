import type { PackedMask } from "@/lib/mask";
import type { Layer } from "@/store/layers";

export function removeDetectionMask(
  prev: Record<number, PackedMask>,
  idx: number
): Record<number, PackedMask> {
  const next: Record<number, PackedMask> = {};
  for (const [k, v] of Object.entries(prev)) {
    const ki = Number(k);
    if (ki < idx) next[ki] = v;
    else if (ki > idx) next[ki - 1] = v;
  }
  return next;
}

export function removeDetectionIndexFromLayers(layers: Layer[], idx: number): Layer[] {
  const next: Layer[] = [];
  for (const l of layers) {
    if (l.detectionIndex === idx) continue;
    next.push(
      l.detectionIndex !== undefined && l.detectionIndex > idx
        ? { ...l, detectionIndex: l.detectionIndex - 1 }
        : l
    );
  }
  return next;
}

export function shiftSelectedDetection(
  selected: number | null,
  idx: number
): number | null {
  if (selected === null || selected === idx) return null;
  return selected > idx ? selected - 1 : selected;
}
