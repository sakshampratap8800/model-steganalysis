import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import { scansApi } from '../services/api';

const MAX_SIZE_BYTES = 1024 * 1024 * 1024; // 1 GB
const ALLOWED_EXTENSIONS = ['.onnx', '.pt', '.pth', '.bin', '.pb'];

export default function UploadScan() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const validateAndSetFile = (selectedFile: File) => {
    const ext = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported file type '${ext}'. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`);
      return false;
    }
    if (selectedFile.size > MAX_SIZE_BYTES) {
      setError(`File exceeds maximum allowed size of 1 GB.`);
      return false;
    }
    setFile(selectedFile);
    setError(null);
    return true;
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setIsUploading(true);
    setError(null);
    
    try {
      const data = await scansApi.create(file);
      if (data.scan_id) {
        navigate(`/scan/${data.scan_id}`);
      } else {
        throw new Error("Invalid response from server");
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An unexpected error occurred during upload.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-100 mb-2">Initiate New Scan</h1>
        <p className="text-slate-400">Upload a model file (.onnx, .pt, .pth) to scan for steganographic payloads and structural anomalies.</p>
      </div>
      
      <div 
        className={clsx(
          "border-2 border-dashed rounded-xl p-12 text-center transition-colors duration-200 ease-in-out cursor-pointer",
          isDragging ? "border-cyber-400 bg-cyber-500/10" : "border-slate-700 hover:border-slate-500 hover:bg-slate-800/50"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input 
          type="file" 
          ref={fileInputRef} 
          onChange={handleFileChange} 
          className="hidden" 
          accept=".onnx,.pt,.pth,.bin,.pb"
        />
        
        <div className="flex flex-col items-center justify-center space-y-4">
          <svg className="w-16 h-16 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <div className="text-lg font-medium text-slate-300">
            {file ? file.name : "Drag and drop your model file here"}
          </div>
          <div className="text-sm text-slate-500">
            {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "Supports .onnx, .pt up to 1GB"}
          </div>
        </div>
      </div>
      
      {error && (
        <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400">
          {error}
        </div>
      )}
      
      <div className="mt-8 flex justify-end">
        <button 
          onClick={handleUpload}
          disabled={!file || isUploading}
          className="px-6 py-3 bg-cyber-600 hover:bg-cyber-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium rounded-lg transition-colors flex items-center space-x-2"
        >
          {isUploading ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>Uploading...</span>
            </>
          ) : (
            <span>Scan Model</span>
          )}
        </button>
      </div>
    </div>
  );
}
