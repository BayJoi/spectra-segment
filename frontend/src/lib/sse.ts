type SseUrl = string | (() => string | Promise<string>);

export function connectSse(
  url: SseUrl,
  onMessage: (data: string) => void,
  onStateChange?: (connected: boolean) => void,
  onOpen?: () => void
): { close: () => void } {
  let es: EventSource | null = null;
  let closed = false;
  let attempt = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const retry = () => {
    if (closed) return;
    const delay = Math.min(30000, 1000 * 2 ** attempt++);
    timer = setTimeout(open, delay);
  };

  const open = async () => {
    if (closed) return;
    let target: string;
    try {
      target = typeof url === "function" ? await url() : url;
    } catch {
      onStateChange?.(false);
      retry();
      return;
    }
    if (closed) return;

    try {
      es = new EventSource(target);
    } catch {
      onStateChange?.(false);
      retry();
      return;
    }

    es.onopen = () => {
      attempt = 0;
      onStateChange?.(true);
      try {
        onOpen?.();
      } catch {
      }
    };
    es.onmessage = (ev) => onMessage(ev.data);
    es.onerror = () => {
      es?.close();
      es = null;
      onStateChange?.(false);
      retry();
    };
  };

  void open();

  return {
    close: () => {
      closed = true;
      if (timer) clearTimeout(timer);
      es?.close();
      onStateChange?.(false);
    },
  };
}
