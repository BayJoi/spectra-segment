export type HistoryKind = "undo" | "redo";

interface HistoryOp {
  objectId: number;
  kind: HistoryKind;
  steps: number;
}

export function availableSteps(depth: number, pendingStrokes: number, kind: HistoryKind): number {
  const usable = kind === "undo" ? depth + pendingStrokes : depth - pendingStrokes;
  return usable > 0 ? usable : 0;
}

export class HistoryQueue {
  private undos = new Map<number, number>();
  private redos = new Map<number, number>();

  push(objectId: number, kind: HistoryKind, steps = 1): void {
    if (steps <= 0) return;
    const map = kind === "undo" ? this.undos : this.redos;
    map.set(objectId, (map.get(objectId) ?? 0) + steps);
  }

  get size(): number {
    return this.undos.size + this.redos.size;
  }

  drain(): HistoryOp[] {
    const ops: HistoryOp[] = [];
    for (const [objectId, steps] of this.undos) ops.push({ objectId, kind: "undo", steps });
    for (const [objectId, steps] of this.redos) ops.push({ objectId, kind: "redo", steps });
    this.undos.clear();
    this.redos.clear();
    return ops;
  }

  clear(): void {
    this.undos.clear();
    this.redos.clear();
  }
}
