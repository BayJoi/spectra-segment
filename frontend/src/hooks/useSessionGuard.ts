import { useEffect, useRef } from "react";
import { useAtom } from "jotai";
import { sessionIdAtom } from "@/store/session";
import { api } from "@/lib/api";

interface UseSessionGuardOptions {
  recoverSession: () => Promise<void>;
}

export function useSessionGuard({ recoverSession }: UseSessionGuardOptions) {
  const [sessionId] = useAtom(sessionIdAtom);

  const sessionIdRef = useRef(sessionId);
  const releasedRef = useRef(false);
  const recoverRef = useRef(recoverSession);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    recoverRef.current = recoverSession;
  }, [recoverSession]);

  useEffect(() => {
    releasedRef.current = false;
  }, [sessionId]);

  useEffect(() => {
    const handlePageHide = () => {
      const sid = sessionIdRef.current;
      if (!sid || releasedRef.current) return;
      releasedRef.current = true;
      api.releaseSession(sid);
    };

    const handlePageShow = (e: PageTransitionEvent) => {
      if (!e.persisted) return;
      const sid = sessionIdRef.current;
      if (!sid) return;
      releasedRef.current = false;
      api.sessionHealth(sid).then((alive) => {
        if (!alive) recoverRef.current();
      });
    };

    window.addEventListener("pagehide", handlePageHide);
    window.addEventListener("pageshow", handlePageShow);
    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      window.removeEventListener("pageshow", handlePageShow);
    };
  }, []);
}
