import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import PromptInput from './components/PromptInput';
import ResponseDisplay from './components/ResponseDisplay';
import VectorDatabaseManager from './components/VectorDatabaseManager';

const App: React.FC = () => {
	const [prompt, setPrompt] = useState<string>('');
	const [response, setResponse] = useState<string>('');
	const [isLoading, setIsLoading] = useState<boolean>(false);
	const [error, setError] = useState<string | null>(null);

	const handleSubmit = useCallback(async () => {
		if (!prompt || isLoading) return;

		setIsLoading(true);
		setError(null);
		setResponse('');

		try {
			// Replace with your actual API endpoint
		} catch (e: unknown) {
			if (e instanceof Error) {
				setError(`An error occurred: ${e.message}`);
			} else {
				setError('An unknown error occurred. Please try again.');
			}
		} finally {
			setIsLoading(false);
		}
	}, [prompt, isLoading]);

	return (
		<div className="min-h-screen min-w-screen bg-slate-900 text-white flex flex-col items-center p-4 sm:p-6 font-sans">
			<div className="w-[80%] flex flex-col h-full">
				<Header />
				<VectorDatabaseManager
					// onBuild={handleBuildKnowledgeBase}
					// isProcessing={isProcessingFiles}
					// status={processingStatus}
					// vectorCount={vectorStore.length}
				/>
				<main className="grow flex flex-col mt-4 bg-slate-800/50 rounded-2xl shadow-2xl overflow-hidden backdrop-blur-sm border border-slate-700">
					<ResponseDisplay
						isLoading={isLoading}
						error={error}
						response={response}
					/>
					<div className="p-4 border-t border-slate-700 bg-slate-900/30">
						<PromptInput
							prompt={prompt}
							setPrompt={setPrompt}
							onSubmit={handleSubmit}
							isLoading={isLoading}
						/>
					</div>
				</main>
				<footer className="text-center py-4 text-slate-500 text-xs">
					<p>For informational purposes only. Not legal advice.</p>
				</footer>
			</div>
		</div>
	);
};

export default App;
