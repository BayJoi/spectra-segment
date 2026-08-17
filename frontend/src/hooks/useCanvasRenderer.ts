import { useCallback, useRef } from "react";
import type { CanvasState, Detection } from "@/lib/types";
import type { Sam3Instance } from "@/store/sam3";
import type { PackedMask } from "@/lib/mask";

function clampIndex(v: number, max: number): number {
  return v < 0 ? 0 : v >= max ? max - 1 : v;
}

function clampByte(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

let ctxFilterSupported: boolean | null = null;

function supportsCtxFilter(): boolean {
  if (ctxFilterSupported === null) {
    try {
      const probe = document.createElement("canvas");
      probe.width = 16;
      probe.height = 16;
      const pctx = probe.getContext("2d");
      if (!pctx) {
        ctxFilterSupported = false;
      } else {
        pctx.filter = "blur(8px)";
        pctx.fillStyle = "#fff";
        pctx.fillRect(4, 4, 8, 8);
        pctx.filter = "none";
        ctxFilterSupported = pctx.getImageData(0, 0, 1, 1).data[3] > 0;
      }
    } catch {
      ctxFilterSupported = false;
    }
  }
  return ctxFilterSupported;
}

function boxBlurFloat(src: Float32Array, w: number, h: number, radius: number): Float32Array {
  const r = Math.max(1, Math.round(radius));
  const size = r * 2 + 1;
  const half = w * h;
  const dst = new Float32Array(half);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let sum = 0;
    for (let k = -r; k <= r; k++) sum += src[row + clampIndex(k, w)];
    dst[row] = sum / size;
    for (let x = 1; x < w; x++) {
      sum += src[row + clampIndex(x + r, w)] - src[row + clampIndex(x - r - 1, w)];
      dst[row + x] = sum / size;
    }
  }
  const out = new Float32Array(half);
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let k = -r; k <= r; k++) sum += dst[clampIndex(k, h) * w + x];
    out[x] = sum / size;
    for (let y = 1; y < h; y++) {
      sum += dst[clampIndex(y + r, h) * w + x] - dst[clampIndex(y - r - 1, h) * w + x];
      out[y * w + x] = sum / size;
    }
  }
  return out;
}

function blurPadded(source: HTMLCanvasElement, radius: number, crop: boolean): HTMLCanvasElement {
  const w = source.width;
  const h = source.height;
  const r = Math.max(1, Math.round(radius));
  const pw = w + r * 2;
  const ph = h + r * 2;

  const padded = document.createElement("canvas");
  padded.width = pw;
  padded.height = ph;
  padded.getContext("2d")!.drawImage(source, r, r);

  let blurred: HTMLCanvasElement;
  if (supportsCtxFilter()) {
    blurred = document.createElement("canvas");
    blurred.width = pw;
    blurred.height = ph;
    const bctx = blurred.getContext("2d")!;
    bctx.filter = `blur(${r}px)`;
    bctx.drawImage(padded, 0, 0);
  } else {
    const srcData = padded.getContext("2d")!.getImageData(0, 0, pw, ph).data;
    const len = srcData.length;
    const premul = new Float32Array(len);
    for (let i = 0; i < len; i += 4) {
      const a = srcData[i + 3];
      const k = a / 255;
      premul[i] = srcData[i] * k;
      premul[i + 1] = srcData[i + 1] * k;
      premul[i + 2] = srcData[i + 2] * k;
      premul[i + 3] = a;
    }
    let blurredF: Float32Array = premul;
    for (let pass = 0; pass < 3; pass++) {
      blurredF = boxBlurFloat(blurredF, pw, ph, r);
    }
    blurred = document.createElement("canvas");
    blurred.width = pw;
    blurred.height = ph;
    const bctx = blurred.getContext("2d")!;
    const bd = bctx.createImageData(pw, ph);
    const dd = bd.data;
    for (let i = 0; i < len; i += 4) {
      const a = blurredF[i + 3];
      if (a > 0) {
        dd[i] = clampByte(Math.round((blurredF[i] / a) * 255));
        dd[i + 1] = clampByte(Math.round((blurredF[i + 1] / a) * 255));
        dd[i + 2] = clampByte(Math.round((blurredF[i + 2] / a) * 255));
        dd[i + 3] = clampByte(Math.round(a));
      }
    }
    bctx.putImageData(bd, 0, 0);
  }

  if (!crop) return blurred;
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const octx = out.getContext("2d")!;
  octx.drawImage(blurred, r, r, w, h, 0, 0, w, h);
  return out;
}

function blurMaskCanvas(source: HTMLCanvasElement, radius: number): HTMLCanvasElement {
  return blurPadded(source, radius, true);
}

interface CanvasRendererRefs {
  containerRef: React.RefObject<HTMLDivElement | null>;
  imageCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  drawCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  maskCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  detectCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  imageRef: React.RefObject<HTMLImageElement | null>;
}

interface RendererState {
  masksRef: React.MutableRefObject<PackedMask[]>;
  detectionsRef: React.MutableRefObject<Detection[]>;
  selectedDetectionRef: React.MutableRefObject<number | null>;
  showTransparentRef: React.MutableRefObject<boolean>;
  hideBboxesRef: React.MutableRefObject<boolean>;
  featherRadiusRef: React.MutableRefObject<number>;
  sam3InstancesRef: React.MutableRefObject<Sam3Instance[]>;
  selectedSam3InstanceRef: React.MutableRefObject<{ promptIndex: number; instanceIndex: number } | null>;
}

export function useCanvasRenderer(
  refs: CanvasRendererRefs,
  state: RendererState,
  zoomRef: React.MutableRefObject<number>,
  panRef: React.MutableRefObject<{ x: number; y: number }>
) {
  const maskCacheRef = useRef<HTMLCanvasElement | null>(null);
  const maskCacheKeyRef = useRef("");
  const transparentCompositeRef = useRef<HTMLCanvasElement | null>(null);
  const transparentCompositeKeyRef = useRef("");
  const checkerPatternCacheRef = useRef<{ pattern: CanvasPattern; width: number; height: number } | null>(null);

  const computeState = useCallback(
    (zoom: number, panX: number, panY: number): CanvasState => {
      const container = refs.containerRef.current;
      const img = refs.imageRef.current;
      if (!container || !img)
        return { imageWidth: 0, imageHeight: 0, canvasWidth: 0, canvasHeight: 0, scale: 1, offsetX: 0, offsetY: 0 };

      const cw = container.clientWidth;
      const ch = container.clientHeight;
      const iw = img.naturalWidth;
      const ih = img.naturalHeight;
      const baseScale = Math.min(cw / iw, ch / ih, 1);
      const scale = baseScale * zoom;
      const ox = Math.floor((cw - Math.floor(iw * scale)) / 2) + panX;
      const oy = Math.floor((ch - Math.floor(ih * scale)) / 2) + panY;

      return { imageWidth: iw, imageHeight: ih, canvasWidth: cw, canvasHeight: ch, scale, offsetX: ox, offsetY: oy };
    },
    [refs.containerRef, refs.imageRef]
  );

  const rebuildMaskCache = useCallback((currentMasks: PackedMask[], featherRadius: number) => {
    const img = refs.imageRef.current;
    if (!img || !currentMasks.length) {
      maskCacheRef.current = null;
      maskCacheKeyRef.current = "";
      return;
    }

    const iw = img.naturalWidth;
    const ih = img.naturalHeight;
    const key = currentMasks.map((m) => m ? m.w : 0).join(",") + `:f${featherRadius}`;
    if (key === maskCacheKeyRef.current && maskCacheRef.current) return;

    const off = document.createElement("canvas");
    off.width = iw;
    off.height = ih;
    const octx = off.getContext("2d")!;
    const imgData = octx.createImageData(iw, ih);
    const u32 = new Uint32Array(imgData.data.buffer, imgData.data.byteOffset, imgData.data.byteLength >>> 2);
    const TINT = 0x783475d4;

    for (const mask of currentMasks) {
      if (!mask || mask.w !== iw || mask.h !== ih) continue;
      const src = mask.data;
      for (let i = 0; i < src.length; i++) {
        if (src[i]) u32[i] = TINT;
      }
    }
    octx.putImageData(imgData, 0, 0);

    if (featherRadius > 0) {
      maskCacheRef.current = blurMaskCanvas(off, featherRadius);
    } else {
      maskCacheRef.current = off;
    }
    maskCacheKeyRef.current = key;
  }, [refs.imageRef]);

  const rebuildTransparentComposite = useCallback((currentMasks: PackedMask[], featherRadius: number) => {
    const img = refs.imageRef.current;
    if (!img || !currentMasks.length) {
      transparentCompositeRef.current = null;
      transparentCompositeKeyRef.current = "";
      return;
    }
    const iw = img.naturalWidth;
    const ih = img.naturalHeight;
    const key = currentMasks.map((m) => m ? m.w : 0).join(",") + `:f${featherRadius}`;
    if (key === transparentCompositeKeyRef.current && transparentCompositeRef.current) return;

    const maskOff = document.createElement("canvas");
    maskOff.width = iw;
    maskOff.height = ih;
    const mctx = maskOff.getContext("2d")!;
    const imgData = mctx.createImageData(iw, ih);
    const u32 = new Uint32Array(imgData.data.buffer, imgData.data.byteOffset, imgData.data.byteLength >>> 2);
    for (const mask of currentMasks) {
      if (!mask || mask.w !== iw || mask.h !== ih) continue;
      const src = mask.data;
      for (let i = 0; i < src.length; i++) {
        if (src[i]) u32[i] = 0xffffffff;
      }
    }
    mctx.putImageData(imgData, 0, 0);

    let maskSource: HTMLCanvasElement = maskOff;
    if (featherRadius > 0) {
      maskSource = blurMaskCanvas(maskOff, featherRadius);
    }

    const off = document.createElement("canvas");
    off.width = iw;
    off.height = ih;
    const octx = off.getContext("2d")!;
    octx.drawImage(img, 0, 0, iw, ih);
    octx.globalCompositeOperation = "destination-in";
    octx.drawImage(maskSource, 0, 0);

    transparentCompositeRef.current = off;
    transparentCompositeKeyRef.current = key;
  }, [refs.imageRef]);

  const renderMasks = useCallback(
    (canvasState: CanvasState) => {
      const canvas = refs.maskCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const currentMasks = state.masksRef.current;
      const img = refs.imageRef.current;
      const transparent = state.showTransparentRef.current;
      const featherRadius = state.featherRadiusRef.current;

      if (transparent) {
        if (!img) return;
        const { scale, offsetX: ox, offsetY: oy } = canvasState;
        const dw = Math.floor(img.naturalWidth * scale);
        const dh = Math.floor(img.naturalHeight * scale);
        if (dw <= 0 || dh <= 0) return;

        if (!currentMasks.length) {
          ctx.drawImage(img, ox, oy, dw, dh);
          return;
        }

        rebuildTransparentComposite(currentMasks, featherRadius);
        if (transparentCompositeRef.current) {
          ctx.drawImage(transparentCompositeRef.current, ox, oy, dw, dh);
        }
      } else {
        rebuildMaskCache(currentMasks, featherRadius);
        if (maskCacheRef.current) {
          ctx.drawImage(maskCacheRef.current, canvasState.offsetX, canvasState.offsetY, Math.floor(img!.naturalWidth * canvasState.scale), Math.floor(img!.naturalHeight * canvasState.scale));
        }
      }
    },
    [refs.maskCanvasRef, refs.imageRef, state.masksRef, state.showTransparentRef, state.featherRadiusRef, rebuildMaskCache, rebuildTransparentComposite]
  );

  const renderDetections = useCallback(
    (canvasState: CanvasState) => {
      const canvas = refs.detectCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d")!;

      const currentDetections = state.detectionsRef.current;
      if (!currentDetections.length) return;

      if (state.hideBboxesRef.current && state.showTransparentRef.current) return;

      const { scale, offsetX: ox, offsetY: oy } = canvasState;
      const currentSelected = state.selectedDetectionRef.current;

      currentDetections.forEach((det: Detection, i: number) => {
        const [x1, y1, x2, y2] = det.bbox;
        const bx = x1 * scale + ox;
        const by = y1 * scale + oy;
        const bw = (x2 - x1) * scale;
        const bh = (y2 - y1) * scale;

        const isSelected = i === currentSelected;

        ctx.strokeStyle = isSelected ? "#f97316" : "#00ff88";
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeRect(bx, by, bw, bh);

        ctx.fillStyle = isSelected ? "rgba(249, 115, 22, 0.15)" : "rgba(0, 255, 136, 0.08)";
        ctx.fillRect(bx, by, bw, bh);

        const label = `#${i + 1} ${det.label} ${Math.round(det.score * 100)}%`;
        ctx.font = "12px JetBrains Mono, monospace";
        const textWidth = ctx.measureText(label).width;
        const labelH = 20;
        const labelW = textWidth + 12;
        const labelY = by - labelH > 0 ? by - labelH : by;

        ctx.fillStyle = isSelected ? "#f97316" : "#111";
        ctx.fillRect(bx, labelY, labelW, labelH);
        ctx.strokeStyle = isSelected ? "#f97316" : "#00ff88";
        ctx.lineWidth = 1;
        ctx.strokeRect(bx, labelY, labelW, labelH);

        ctx.fillStyle = isSelected ? "#fff" : "#00ff88";
        ctx.textBaseline = "middle";
        ctx.fillText(label, bx + 6, labelY + labelH / 2);
      });
    },
    [refs.detectCanvasRef, state.detectionsRef, state.selectedDetectionRef]
  );

  const renderSam3Boxes = useCallback(
    (canvasState: CanvasState) => {
      const canvas = refs.detectCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d")!;

      const instances = state.sam3InstancesRef.current;
      if (!instances.length) return;

      if (state.hideBboxesRef.current && state.showTransparentRef.current) return;

      const { scale, offsetX: ox, offsetY: oy } = canvasState;
      const selected = state.selectedSam3InstanceRef.current;

      instances.forEach((inst) => {
        if (!inst.bbox || inst.bbox.length < 4) return;
        const [x1, y1, x2, y2] = inst.bbox;
        const bx = x1 * scale + ox;
        const by = y1 * scale + oy;
        const bw = (x2 - x1) * scale;
        const bh = (y2 - y1) * scale;

        const isSelected =
          selected?.promptIndex === inst.promptIndex && selected?.instanceIndex === inst.instanceIndex;

        ctx.strokeStyle = isSelected ? "#f97316" : "#00ff88";
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeRect(bx, by, bw, bh);

        ctx.fillStyle = isSelected ? "rgba(249, 115, 22, 0.15)" : "rgba(0, 255, 136, 0.08)";
        ctx.fillRect(bx, by, bw, bh);

        const label = `#${inst.instanceIndex + 1} ${inst.text}`;
        ctx.font = "12px JetBrains Mono, monospace";
        const textWidth = ctx.measureText(label).width;
        const labelH = 20;
        const labelW = textWidth + 12;
        const labelY = by - labelH > 0 ? by - labelH : by;

        ctx.fillStyle = isSelected ? "#f97316" : "#111";
        ctx.fillRect(bx, labelY, labelW, labelH);
        ctx.strokeStyle = isSelected ? "#f97316" : "#00ff88";
        ctx.lineWidth = 1;
        ctx.strokeRect(bx, labelY, labelW, labelH);

        ctx.fillStyle = isSelected ? "#fff" : "#00ff88";
        ctx.textBaseline = "middle";
        ctx.fillText(label, bx + 6, labelY + labelH / 2);
      });
    },
    [refs.detectCanvasRef, state.sam3InstancesRef, state.selectedSam3InstanceRef, state.hideBboxesRef, state.showTransparentRef]
  );

  const redraw = useCallback(() => {
    const container = refs.containerRef.current;
    const img = refs.imageRef.current;
    if (!container || !img) return;

    const canvasState = computeState(zoomRef.current, panRef.current.x, panRef.current.y);
    const { scale, offsetX: ox, offsetY: oy } = canvasState;
    const w = Math.floor(img.naturalWidth * scale);
    const h = Math.floor(img.naturalHeight * scale);

    const imgCanvas = refs.imageCanvasRef.current;
    if (imgCanvas) {
      imgCanvas.width = canvasState.canvasWidth;
      imgCanvas.height = canvasState.canvasHeight;
      const ctx = imgCanvas.getContext("2d")!;
      ctx.clearRect(0, 0, canvasState.canvasWidth, canvasState.canvasHeight);

      if (state.showTransparentRef.current) {
        const sz = 16;
        const cw = canvasState.canvasWidth;
        const ch = canvasState.canvasHeight;
        if (!checkerPatternCacheRef.current || checkerPatternCacheRef.current.width !== cw || checkerPatternCacheRef.current.height !== ch) {
          const tile = document.createElement("canvas");
          tile.width = sz * 2;
          tile.height = sz * 2;
          const tctx = tile.getContext("2d")!;
          tctx.fillStyle = "#1a1a1a";
          tctx.fillRect(0, 0, sz * 2, sz * 2);
          tctx.fillStyle = "#222";
          tctx.fillRect(sz, 0, sz, sz);
          tctx.fillRect(0, sz, sz, sz);
          const pattern = ctx.createPattern(tile, "repeat")!;
          checkerPatternCacheRef.current = { pattern, width: cw, height: ch };
        }
        ctx.fillStyle = checkerPatternCacheRef.current.pattern;
        ctx.fillRect(0, 0, cw, ch);
      } else {
        ctx.drawImage(img, ox, oy, w, h);
      }
    }

    for (const ref of [refs.drawCanvasRef, refs.maskCanvasRef, refs.detectCanvasRef]) {
      if (ref.current) {
        ref.current.width = canvasState.canvasWidth;
        ref.current.height = canvasState.canvasHeight;
      }
    }

    renderMasks(canvasState);
    renderDetections(canvasState);
    renderSam3Boxes(canvasState);
  }, [refs, computeState, renderMasks, renderDetections, renderSam3Boxes, state.showTransparentRef, zoomRef, panRef]);

  const invalidateCache = useCallback(() => {
    maskCacheKeyRef.current = "";
    transparentCompositeKeyRef.current = "";
    checkerPatternCacheRef.current = null;
  }, []);

  return { computeState, redraw, invalidateCache };
}
