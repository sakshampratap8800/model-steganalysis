import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import UploadScan from "./pages/UploadScan";
import ScanDetails from "./pages/ScanDetails";
import ScanHistory from "./pages/ScanHistory";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/scan/new" element={<UploadScan />} />
        <Route path="/scan/:scanId" element={<ScanDetails />} />
        <Route path="/history" element={<ScanHistory />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
