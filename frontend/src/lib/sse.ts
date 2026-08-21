export interface SseHandle {
  close: () => void;
}

export function connectSse(
  url: string,
  onMessage: (data: string) => void,
  onStateChange?: (connected: boolean) => void
): SseHandle {
  let es: EventSource | null = null;
  let closed = false;
  let attempt = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const open = () => {
    if (closed) return;
    es = new EventSource(url);
    es.onopen = () => {
      attempt = 0;
      onStateChange?.(true);
    };
    es.onmessage = (ev) => onMessage(ev.data);
    es.onerror = () => {
      es?.close();
      es = null;
      onStateChange?.(false);
      if (closed) return;
      const delay = Math.min(30000, 1000 * 2 ** attempt++);
      timer = setTimeout(open, delay);
    };
  };

  open();

  return {
    close: () => {
      closed = true;
      if (timer) clearTimeout(timer);
      es?.close();
      onStateChange?.(false);
    },
  };
}
