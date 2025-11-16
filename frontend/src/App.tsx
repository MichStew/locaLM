import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import PromptInput from './components/PromptInput';
import ResponseDisplay from './components/ResponseDisplay';

const API_URL =
	import.meta.env.VITE_API_URL || 'http://localhost:5000/api/ask';

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
			const res = await fetch(API_URL, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ question: prompt }),
			});

			if (!res.ok) {
				const text = await res.text();
				throw new Error(
					`API error (${res.status} ${res.statusText}): ${text}`,
				);
			}

			const data = (await res.json()) as { answer?: string; error?: string };
			if (data.error) {
				throw new Error(data.error);
			}
			setResponse(data.answer?.trim() || 'No answer returned.');
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
			<div className="w-full flex flex-col h-full">
				<Header />
				<main className="flex-grow flex flex-col bg-slate-800/50 rounded-2xl shadow-2xl overflow-hidden backdrop-blur-sm border border-slate-700">
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
					<p>
						For informational purposes only. Not
						legal advice.
					</p>
				</footer>
			</div>
		</div>
	);
};

export default App;
