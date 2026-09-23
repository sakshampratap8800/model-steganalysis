import { useState, useEffect } from 'react';
import * as Comlink from 'comlink';

// Assuming scanner.worker.ts exposes init() and scanFile()
export function useWasmScanner() {
  const [isReady, setIsReady] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [workerApi, setWorkerApi] = useState<any>(null);

  useEffect(() => {
    // Initialize Web Worker
    const worker = new Worker(new URL('../workers/scanner.worker.ts', import.meta.url), { type: 'module' });
    const api = Comlink.wrap(worker) as any;
    
    api.init().then(() => {
      setWorkerApi(api);
      setIsReady(true);
    });

    return () => {
      worker.terminate();
    };
  }, []);

  const scan = async (file: File) => {
    if (!workerApi) throw new Error("Worker not initialized");
    setIsScanning(true);
    try {
      const arrayBuffer = await file.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      // In a real implementation, you'd extract meta JSON before sending
      const result = await workerApi.scanFile("{}", uint8Array);
      return result;
    } finally {
      setIsScanning(false);
    }
  };

  return { scan, isReady, isScanning };
}
