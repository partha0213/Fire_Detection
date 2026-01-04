'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, Bell, Settings, Camera, Flame, ChevronRight } from 'lucide-react';
import { Header } from '../components/Header';
import { DetectionResults } from '../components/DetectionResults';
import { MonitorView } from '../components/MonitorView';
import { AlertsView } from '../components/AlertsView';
import { ProcessingView } from '../components/ProcessingView';
import { DetectionResult, Alert, AppTab, ApiStatus } from '../types';

export default function Home() {
    const [activeTab, setActiveTab] = useState<AppTab>('detect');
    const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [apiStatus, setApiStatus] = useState<ApiStatus>({ status: 'checking', latency: 0 });
    const [error, setError] = useState<string | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Initial check
    useEffect(() => {
        checkHealth();
    }, []);

    const checkHealth = useCallback(async () => {
        setApiStatus(prev => ({ ...prev, status: 'checking' }));
        const start = Date.now();
        try {
            const response = await fetch('http://localhost:8000/health', { signal: AbortSignal.timeout(5000) });
            const latency = Date.now() - start;
            if (response.ok) {
                const data = await response.json();
                setApiStatus({ status: 'online', latency });
                console.log('API Status:', data);
            } else {
                throw new Error('Health check failed');
            }
        } catch {
            setApiStatus({ status: 'offline', latency: 0 });
        }
    }, []);

    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => setSelectedImage(e.target?.result as string);
        reader.readAsDataURL(file);

        setIsLoading(true);
        setDetectionResult(null);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', file);
            const response = await fetch('http://localhost:8000/detect', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                // Map API response to frontend format
                setDetectionResult({
                    fire_detected: result.fire_detected,
                    confidence: result.confidence,
                    fire_type: result.fire_type,
                    fire_type_probs: result.fire_type_probs,
                    timestamp: result.timestamp,
                    bounding_boxes: result.detections?.map((d: any) => d.bbox) || [],
                    attention_weights: result.fusion_weights || {
                        yolo: 0.50,
                        vit: 0.30,
                        optical_flow: 0.20,
                    }
                });
            } else {
                const errData = await response.json();
                setError(errData.detail || 'Detection failed');
            }
        } catch (err) {
            console.error('Error:', err);
            setError('Failed to connect to API. Make sure the backend is running.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file && fileInputRef.current) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInputRef.current.files = dataTransfer.files;
            handleFileSelect({ target: { files: dataTransfer.files } } as any);
        }
    };

    const fetchAlerts = async () => {
        try {
            const response = await fetch('http://localhost:8000/alerts');
            if (response.ok) {
                const data = await response.json();
                setAlerts(data.alerts || []);
            }
        } catch (err) {
            console.error('Error fetching alerts:', err);
        }
    };

    useEffect(() => {
        if (activeTab === 'alerts') fetchAlerts();
    }, [activeTab]);

    return (
        <div className="min-h-screen pb-20 selection:bg-orange-500/30">
            <Header apiStatus={apiStatus} checkHealth={checkHealth} />

            <div className="max-w-7xl mx-auto px-6">

                {/* API Offline Banner */}
                {apiStatus.status === 'offline' && (
                    <div className="mb-6 p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center justify-between animate-in fade-in slide-in-from-top-2">
                        <span className="text-sm text-red-200">
                            <strong>API Offline:</strong> Backend server is not running.
                        </span>
                        <button
                            onClick={checkHealth}
                            className="text-xs text-red-400 hover:text-red-300 underline"
                        >
                            Retry Connection
                        </button>
                    </div>
                )}

                {/* Error Banner */}
                {error && (
                    <div className="mb-6 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg flex items-center justify-between animate-in fade-in slide-in-from-top-2">
                        <span className="text-sm text-yellow-200">
                            <strong>Error:</strong> {error}
                        </span>
                        <button
                            onClick={() => setError(null)}
                            className="text-xs text-yellow-400 hover:text-yellow-300 underline"
                        >
                            Dismiss
                        </button>
                    </div>
                )}

                {/* Nav Tabs */}
                <div className="flex p-1 bg-slate-900/50 backdrop-blur rounded-xl border border-white/5 w-fit mb-8 mx-auto lg:mx-0">
                    {[
                        { id: 'detect', label: 'Analysis', icon: Upload },
                        { id: 'monitor', label: 'Live Monitor', icon: Camera },
                        { id: 'alerts', label: 'History', icon: Bell },
                        { id: 'settings', label: 'Config', icon: Settings },
                    ].map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as AppTab)}
                            disabled={isLoading}
                            className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 ${activeTab === tab.id
                                ? 'bg-gradient-to-r from-orange-600 to-red-600 text-white shadow-lg shadow-orange-900/20'
                                : 'text-slate-400 hover:text-white hover:bg-white/5 disabled:opacity-50'
                                }`}
                        >
                            <tab.icon className="w-4 h-4" />
                            {tab.label}
                        </button>
                    ))}
                </div>

                <main className="min-h-[600px] relative">
                    {/* Background decoration */}
                    <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none -z-10">
                        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-orange-600/5 rounded-full blur-3xl"></div>
                        <div className="absolute bottom-1/4 left-1/4 w-96 h-96 bg-blue-600/5 rounded-full blur-3xl"></div>
                    </div>

                    {activeTab === 'detect' && (
                        isLoading && selectedImage ? (
                            <ProcessingView image={selectedImage} />
                        ) : (
                            <div className="grid lg:grid-cols-5 gap-8 animate-in fade-in duration-500">
                                {/* Upload Area */}
                                <div className={`${detectionResult ? 'lg:col-span-2' : 'lg:col-span-5 max-w-2xl mx-auto w-full'}`}>
                                    <div className={`glass-panel rounded-2xl p-8 sticky top-32 transition-all duration-500 ${detectionResult ? '' : 'py-20'}`}>
                                        <div className="text-center mb-8">
                                            <div className="w-16 h-16 bg-slate-800/50 rounded-2xl flex items-center justify-center mx-auto mb-4 border border-white/10 shadow-inner group">
                                                <Flame className="w-8 h-8 text-orange-500 group-hover:scale-110 transition-transform duration-300" />
                                            </div>
                                            <h2 className="text-xl font-bold text-white">Upload Source Data</h2>
                                            <p className="text-slate-400 text-sm mt-2 max-w-xs mx-auto">Upload CCTV frames or thermal imagery for YOLO analysis.</p>
                                        </div>

                                        <div
                                            className="relative group cursor-pointer border-2 border-dashed border-slate-700 hover:border-orange-500/50 hover:bg-slate-800/50 rounded-xl p-8 transition-all duration-300"
                                            onClick={() => fileInputRef.current?.click()}
                                            onDragOver={(e) => e.preventDefault()}
                                            onDrop={handleDrop}
                                        >
                                            <input
                                                ref={fileInputRef}
                                                type="file"
                                                accept="image/*"
                                                onChange={handleFileSelect}
                                                className="hidden"
                                            />

                                            <div className="flex flex-col items-center py-4">
                                                <div className="w-12 h-12 bg-slate-800 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform border border-white/5">
                                                    <Upload className="w-5 h-5 text-slate-400 group-hover:text-white" />
                                                </div>
                                                <p className="text-slate-300 font-medium">Click to upload or drag & drop</p>
                                                <p className="text-slate-500 text-xs mt-2 font-mono">JPG, PNG, WEBP • MAX 10MB</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Results Area */}
                                {detectionResult && (
                                    <div className="lg:col-span-3">
                                        <DetectionResults result={detectionResult} imageUrl={selectedImage} />
                                    </div>
                                )}
                            </div>
                        )
                    )}

                    {activeTab === 'monitor' && <MonitorView />}
                    {activeTab === 'alerts' && <AlertsView alerts={alerts} />}

                    {activeTab === 'settings' && (
                        <div className="max-w-2xl mx-auto glass-panel rounded-2xl p-8 border border-white/5 animate-in fade-in slide-in-from-bottom-4">
                            <h2 className="text-2xl font-bold text-white mb-6">System Configuration</h2>

                            <div className="space-y-8">
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <label className="text-sm font-medium text-slate-300">Device Location Identifier</label>
                                        <span className="text-xs text-orange-400">Required</span>
                                    </div>
                                    <input type="text" className="w-full bg-slate-950/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:ring-1 focus:ring-orange-500 focus:border-orange-500 outline-none transition-all placeholder:text-slate-600" placeholder="e.g. Warehouse Sector 7" />
                                </div>

                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <label className="text-sm font-medium text-slate-300">Detection Confidence Threshold</label>
                                        <span className="text-xs font-mono text-slate-400 bg-slate-800 px-2 py-1 rounded">75%</span>
                                    </div>
                                    <input type="range" className="w-full accent-orange-500 h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer" min="0" max="100" defaultValue="75" />
                                    <div className="flex justify-between text-xs text-slate-500 font-mono">
                                        <span>SENSITIVE</span>
                                        <span>BALANCED</span>
                                        <span>STRICT</span>
                                    </div>
                                </div>

                                <div className="pt-6 border-t border-white/5">
                                    <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500 mb-4">Notification Channels</h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        {['SMS Alerts', 'Email Reports', 'Sound Alarm', 'Auto-call Emergency'].map((item, i) => (
                                            <label key={i} className={`flex items-center justify-between p-4 rounded-xl border cursor-pointer transition-all ${i < 2 ? 'bg-orange-500/5 border-orange-500/20' : 'bg-slate-900/50 border-white/5 hover:border-white/10'}`}>
                                                <span className={i < 2 ? 'text-orange-100' : 'text-slate-400'}>{item}</span>
                                                <div className={`w-5 h-5 rounded-md flex items-center justify-center border ${i < 2 ? 'bg-orange-500 border-orange-500' : 'border-slate-600'}`}>
                                                    {i < 2 && <ChevronRight className="w-3 h-3 text-white rotate-90" />}
                                                </div>
                                            </label>
                                        ))}
                                    </div>
                                </div>

                                <div className="flex justify-end pt-4">
                                    <button className="px-8 py-3 bg-white text-slate-950 font-bold rounded-xl hover:bg-slate-200 transition-colors shadow-lg shadow-white/10">
                                        Save Configuration
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}
