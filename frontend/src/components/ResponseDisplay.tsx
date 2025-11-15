import React, { useState, useEffect } from 'react';

interface ResponseDisplayProps {
	isLoading: boolean;
	error: string | null;
	response: string; // Changed from object to string
}

const ResponseDisplay: React.FC<ResponseDisplayProps> = ({
	isLoading,
	error,
	response,
}) => {
	const [displayedText, setDisplayedText] = useState('');
	const [showCursor, setShowCursor] = useState(false);
	const [showSources, setShowSources] = useState(false); // For future use if sources are added

	useEffect(() => {
		if (!response) {
			setDisplayedText('');
			setShowCursor(false);
			setShowSources(false);
			return;
		}

		setDisplayedText('');
		setShowCursor(true);
		setShowSources(false);

		const textToType = response;

		const interval = setInterval(() => {
			setDisplayedText((currentText) => {
				if (currentText.length === textToType.length) {
					clearInterval(interval);
					setShowCursor(false);
					// A small delay before showing sources for a smoother effect
					setTimeout(() => setShowSources(true), 300);
					return currentText;
				}
				return textToType.substring(0, currentText.length + 1);
			});
		}, 1); // Typing speed in ms

		return () => clearInterval(interval);
	}, [response]);

	const renderContent = () => {
		if (isLoading) {
			return (
				<div className="flex flex-col items-center justify-center h-full text-slate-400">
					<div className="w-8 h-8 border-4 border-slate-500 border-t-cyan-400 rounded-full animate-spin mb-4"></div>
					<p className="text-lg">Generating response...</p>
					<p className="text-sm">
						Please wait while LocaLM analyzes your query.
					</p>
				</div>
			);
		}

		if (error) {
			return (
				<div className="p-6 text-red-400 bg-red-900/20 rounded-lg m-4">
					<h3 className="font-bold mb-2">Error</h3>
					<p>{error}</p>
				</div>
			);
		}

		if (response) {
			// Basic markdown-like formatting.
			const formattedResponse = displayedText
				.split('\n')
				.map((line, index) => {
					if (line.startsWith('### ')) {
						return (
							<h3
								key={index}
								className="text-xl font-semibold mt-4 mb-2 text-cyan-300"
							>
								{line.substring(4)}
							</h3>
						);
					}
					if (line.startsWith('## ')) {
						return (
							<h2
								key={index}
								className="text-2xl font-bold mt-6 mb-3 text-cyan-200 border-b border-slate-700 pb-2"
							>
								{line.substring(3)}
							</h2>
						);
					}
					if (line.startsWith('# ')) {
						return (
							<h1
								key={index}
								className="text-3xl font-extrabold mt-8 mb-4 text-cyan-100"
							>
								{line.substring(2)}
							</h1>
						);
					}
					if (
						line.trim().startsWith('* ') ||
						line.trim().startsWith('- ')
					) {
						return (
							<li key={index} className="ml-6 list-disc">
								{line.substring(line.indexOf(' ') + 1)}
							</li>
						);
					}
					if (line.trim() === '') return null;
					return (
						<p key={index} className="my-2">
							{line}
						</p>
					);
				})
				.filter(Boolean); // Filter out nulls for empty lines

			return (
				<>
					<div className="prose prose-invert prose-p:text-slate-300 prose-headings:text-cyan-200 p-8 min-h-[5rem]">
						{formattedResponse}
						{showCursor && (
							<span className="inline-block w-0.5 h-5 bg-cyan-300 animate-pulse ml-1 align-bottom"></span>
						)}
					</div>
					{/* {response.sources && response.sources.length > 0 && (
						<div
							className={`p-8 pt-0 mt-4 border-t border-slate-700 transition-opacity duration-500 ${showSources ? 'opacity-100' : 'opacity-0'}`}
						>
							<h4 className="text-sm font-bold text-slate-400 mb-2">
								Sources:
							</h4>
							<ul className="list-disc pl-5 text-sm space-y-1">
								{response.sources.map((source, i) => (
									<li
										key={i}
										className="text-slate-400 truncate"
									>
										{source}
									</li>
								))}
							</ul>
						</div>
					)} */}
				</>
			);
		}

		return (
			<div className="flex flex-col items-center justify-center h-full text-slate-500 p-8 text-center">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					className="h-16 w-16 mb-4 opacity-30"
					fill="none"
					viewBox="0 0 24 24"
					stroke="currentColor"
				>
					<path
						strokeLinecap="round"
						strokeLinejoin="round"
						strokeWidth={2}
						d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
					/>
				</svg>
				<h2 className="text-2xl font-semibold text-slate-400">
					Welcome to LocaLM
				</h2>
				<p className="mt-2 max-w-md">
					Your legal insights will appear here. Enter a query below to
					get started.
				</p>
			</div>
		);
	};

	return (
		<div
			className={`p-6 overflow-y-scroll ${response ? 'max-h-[55vh]' : 'max-h-[35vh]'}`}
		>
			{renderContent()}
		</div>
	);
};

export default ResponseDisplay;
