import { useEffect, useRef, useState } from "react";
import { useAtom } from "jotai";
import { imageEncodingAtom, encodingMessageAtom } from "@/store/ui";

const EASE = "cubic-bezier(0.16, 1, 0.3, 1)";
const SWEEP_MS = 850;
const BACKDROP_FADE_MS = 350;

export function EncodingOverlay() {
  const [encoding] = useAtom(imageEncodingAtom);
  const [message] = useAtom(encodingMessageAtom);
  const [sweeping, setSweeping] = useState(false);
  const [blocking, setBlocking] = useState(false);
  const prevEncodingRef = useRef(false);
  const sweepTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const blockTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (sweepTimerRef.current) {
      clearTimeout(sweepTimerRef.current);
      sweepTimerRef.current = null;
    }
    if (blockTimerRef.current) {
      clearTimeout(blockTimerRef.current);
      blockTimerRef.current = null;
    }

    if (encoding) {
      setSweeping(false);
      setBlocking(true);
    } else if (prevEncodingRef.current) {
      setSweeping(true);
      setBlocking(true);
      sweepTimerRef.current = setTimeout(() => {
        setSweeping(false);
        blockTimerRef.current = setTimeout(() => setBlocking(false), 100);
      }, SWEEP_MS);
    } else {
      setBlocking(false);
    }
    prevEncodingRef.current = encoding;
  }, [encoding]);

  useEffect(() => {
    return () => {
      if (sweepTimerRef.current) clearTimeout(sweepTimerRef.current);
      if (blockTimerRef.current) clearTimeout(blockTimerRef.current);
    };
  }, []);

  const show = encoding || sweeping;

  return (
    <div
      role="status"
      aria-live="polite"
      aria-hidden={!show}
      className={`fixed inset-0 z-[70] flex items-center justify-center transition-opacity duration-150 ${show ? "opacity-100" : "opacity-0"} ${blocking ? "" : "pointer-events-none"}`}
      style={{ transitionTimingFunction: EASE }}
    >
      {(encoding || sweeping) && (
        <div
          className="absolute inset-0 bg-black/70 backdrop-blur-lg"
          style={{
            opacity: encoding ? 1 : 0,
            transition: `opacity ${BACKDROP_FADE_MS}ms ${EASE}`,
          }}
        />
      )}

      {sweeping && (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div
            className="absolute inset-y-0 left-0 w-[45vw] animate-screen-sweep mix-blend-screen"
            style={{
              background:
                "linear-gradient(100deg, transparent 0%, rgba(255,255,255,0.05) 30%, rgba(255,255,255,0.16) 50%, rgba(255,255,255,0.05) 70%, transparent 100%), linear-gradient(100deg, transparent 0%, rgba(184,92,42,0) 38%, rgba(184,92,42,0.28) 50%, rgba(184,92,42,0) 62%, transparent 100%)",
            }}
          />
        </div>
      )}

      <div
        className={`relative w-full max-w-sm mx-4 transition-all duration-300 ${encoding ? "scale-100 translate-y-0 opacity-100" : "scale-105 translate-y-0 opacity-0"}`}
        style={{ transitionTimingFunction: EASE }}
      >
        <div className="relative bg-[#0c0c0c]/95 border border-neutral-800/60 rounded-2xl overflow-hidden shadow-2xl shadow-black/60 grain-bg">
          <div className="h-px bg-gradient-to-r from-transparent via-orange-500/30 to-transparent" />

          <div className="p-7 flex flex-col items-center gap-5">
            <div
              className="w-12 h-12 rounded-full animate-spin shrink-0"
              style={{
                animationDuration: "1.4s",
                background:
                  "conic-gradient(from 0deg, transparent 0 20%, rgba(184,92,42,0.35) 55%, #f97316 85%, #fff7ed 100%)",
                WebkitMask: "radial-gradient(farthest-side, transparent calc(100% - 3px), #000 calc(100% - 2.5px))",
                mask: "radial-gradient(farthest-side, transparent calc(100% - 3px), #000 calc(100% - 2.5px))",
              }}
            />

            <div className="text-center">
              <h2 className="text-sm font-semibold text-neutral-100 font-sans animate-pulse-glow">
                {message ?? "Encoding image..."}
              </h2>
              <p className="text-[11px] text-neutral-500 font-sans mt-1.5 leading-relaxed">
                {message
                  ? "Large image is being fitted to the working resolution."
                  : "Segmenting unlocks automatically once the image is ready."}
              </p>
            </div>
          </div>

          <div className="relative h-0.5 w-full bg-neutral-900 overflow-hidden">
            <div className="animate-shimmer-bar" />
          </div>
        </div>
      </div>
    </div>
  );
}
