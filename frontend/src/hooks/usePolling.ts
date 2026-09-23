import { useState, useEffect, useRef, useCallback } from "react";

interface PollingOptions<T> {
  enabled: boolean;
  onSuccess?: (data: T) => void;
  onError?: (err: Error) => void;
  /** Return true to stop polling */
  stopCondition?: (data: T) => boolean;
}

interface PollingResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Custom hook for polling an async function at a given interval.
 *
 * - Immediately fires once when enabled becomes true.
 * - Stops automatically when stopCondition returns true.
 * - Cleans up the interval on unmount.
 */
export function usePolling<T>(
  fn: () => Promise<T>,
  intervalMs: number,
  options: PollingOptions<T>
): PollingResult<T> {
  const { enabled, onSuccess, onError, stopCondition } = options;

  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  // Keep latest callback refs to avoid stale closure issues
  const fnRef = useRef(fn);
  const onSuccessRef = useRef(onSuccess);
  const onErrorRef = useRef(onError);
  const stopConditionRef = useRef(stopCondition);

  useEffect(() => { fnRef.current = fn; }, [fn]);
  useEffect(() => { onSuccessRef.current = onSuccess; }, [onSuccess]);
  useEffect(() => { onErrorRef.current = onError; }, [onError]);
  useEffect(() => { stopConditionRef.current = stopCondition; }, [stopCondition]);

  const stoppedRef = useRef(false);
  const inflightRef = useRef(false);

  const execute = useCallback(async () => {
    if (stoppedRef.current) return;
    if (inflightRef.current) return;  // skip if previous request still pending
    inflightRef.current = true;
    setLoading(true);
    try {
      const result = await fnRef.current();
      setData(result);
      setError(null);
      onSuccessRef.current?.(result);
      if (stopConditionRef.current?.(result)) {
        stoppedRef.current = true;
      }
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err));
      setError(e);
      onErrorRef.current?.(e);
    } finally {
      setLoading(false);
      inflightRef.current = false;
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    stoppedRef.current = false;

    // Immediate first call
    void execute();

    const timer = setInterval(() => {
      if (stoppedRef.current) {
        clearInterval(timer);
        return;
      }
      void execute();
    }, intervalMs);

    return () => {
      clearInterval(timer);
    };
  }, [enabled, intervalMs, execute]);

  return { data, loading, error, refetch: execute };
}
