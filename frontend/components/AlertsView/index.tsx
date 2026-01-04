'use client';

import React from 'react';
import { AlertTriangle, Calendar, CheckCircle2, MapPin, MoreVertical } from 'lucide-react';
import { Alert } from '../../types';

interface AlertsViewProps {
    alerts: Alert[];
}

export const AlertsView: React.FC<AlertsViewProps> = ({ alerts }) => {
    return (
        <div className="max-w-4xl mx-auto space-y-6">
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h2 className="text-2xl font-bold text-white">Alert History</h2>
                    <p className="text-slate-400">Recent incidents and automated flags</p>
                </div>
                <div className="flex gap-2">
                    <select className="bg-slate-900 border border-white/10 text-slate-300 text-sm rounded-lg px-4 py-2 outline-none focus:border-orange-500">
                        <option>All Alerts</option>
                        <option>Critical</option>
                        <option>Resolved</option>
                    </select>
                </div>
            </div>

            {alerts.length === 0 ? (
                <div className="text-center py-20 bg-slate-900/30 rounded-2xl border border-white/5 border-dashed">
                    <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4">
                        <CheckCircle2 className="w-8 h-8 text-emerald-500" />
                    </div>
                    <h3 className="text-white font-medium">All Clear</h3>
                    <p className="text-slate-500 mt-1">No alerts recorded in the system log.</p>
                </div>
            ) : (
                <div className="space-y-4">
                    {alerts.map((alert) => (
                        <div key={alert.id} className="glass-panel glass-panel-hover rounded-xl p-5 border border-white/5 transition-all group">
                            <div className="flex items-start justify-between">
                                <div className="flex items-start gap-4">
                                    <div className={`p-3 rounded-xl ${alert.status === 'resolved'
                                            ? 'bg-emerald-500/10 text-emerald-400'
                                            : 'bg-red-500/10 text-red-400'
                                        }`}>
                                        <AlertTriangle className="w-6 h-6" />
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-3 mb-1">
                                            <h3 className="font-semibold text-white">{alert.fire_type} Detected</h3>
                                            <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${alert.status === 'new' ? 'bg-red-500 text-white' :
                                                    alert.status === 'resolved' ? 'bg-emerald-500/20 text-emerald-400' :
                                                        'bg-yellow-500/20 text-yellow-400'
                                                }`}>
                                                {alert.status}
                                            </span>
                                        </div>

                                        <div className="flex items-center gap-4 text-sm text-slate-400">
                                            <div className="flex items-center gap-1.5">
                                                <MapPin className="w-3.5 h-3.5" />
                                                {alert.location}
                                            </div>
                                            <div className="flex items-center gap-1.5">
                                                <Calendar className="w-3.5 h-3.5" />
                                                {new Date(alert.timestamp).toLocaleString()}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className="text-right">
                                    <div className="text-lg font-bold text-white">
                                        {(alert.confidence * 100).toFixed(0)}%
                                    </div>
                                    <div className="text-xs text-slate-500">Confidence</div>

                                    <button className="mt-2 p-1 hover:bg-white/10 rounded text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <MoreVertical className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
