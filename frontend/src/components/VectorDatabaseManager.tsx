// import React, { useState, useRef } from 'react';
import LoadingSpinner from './icons/LoadingSpinner';

interface VectorDatabaseManagerProps {
    onBuild?: (files: File[]) => void;
    isProcessing?: boolean;
    status?: string;
    vectorCount?: number;
}

const VectorDatabaseManager: React.FC<VectorDatabaseManagerProps> = ({ onBuild, isProcessing, status, vectorCount }) => {
    // const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
    // const fileInputRef = useRef<HTMLInputElement>(null);

    // const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    //     if (event.target.files) {
    //         setSelectedFiles(Array.from(event.target.files));
    //     }
    // };

    // const handleBuildClick = () => {
    //     onBuild(selectedFiles);
    //     setSelectedFiles([]);
    //     if (fileInputRef.current) {
    //         fileInputRef.current.value = '';
    //     }
    // };

    return (
        <div className="bg-slate-800/50 rounded-2xl p-4 border border-slate-700">
            <h2 className="text-lg font-bold text-cyan-300 mb-2">Knowledge Base</h2>
            <p className="text-sm text-slate-400 mb-4">Upload PDF documents to create a searchable knowledge base for context-aware answers.</p>
            
            <div className="flex flex-col sm:flex-row gap-4">
                <div className="flex-grow">
                    <label htmlFor="pdf-upload" className="sr-only">Choose files</label>
                    <input 
                        // ref={fileInputRef}
                        type="file"
                        id="pdf-upload"
                        multiple
                        accept=".pdf"
                        // onChange={handleFileChange}
                        disabled={isProcessing}
                        className="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-cyan-600/20 file:text-cyan-300 hover:file:bg-cyan-600/30 transition"
                    />
                </div>
                <button
                    // onClick={handleBuildClick}
                    // disabled={isProcessing || selectedFiles.length === 0}
                    className="flex items-center justify-center px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-cyan-500"
                >
                    {isProcessing ? <LoadingSpinner /> : <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor"><path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" /><path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" /></svg>}
                    <span>{isProcessing ? 'Building...' : 'Build Knowledge Base'}</span>
                </button>
            </div>

            {/* {(isProcessing || vectorCount > 0 || status) && (
                 <div className="mt-4 text-xs text-slate-400 bg-slate-900/50 p-3 rounded-lg">
                    <p>
                        <span className="font-semibold">Status: </span> 
                        {isProcessing ? status : `Ready. ${vectorCount} documents in knowledge base.`}
                    </p>
                </div>
            )} */}
        </div>
    );
};

export default VectorDatabaseManager;
