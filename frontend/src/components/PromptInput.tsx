import React from 'react';
import LoadingSpinner from './icons/LoadingSpinner';
import PaperAirplaneIcon from './icons/PaperAirplaneIcon';

interface PromptInputProps {
	prompt: string;
	setPrompt: (prompt: string) => void;
	onSubmit: () => void;
	isLoading: boolean;
}

const PromptInput: React.FC<PromptInputProps> = ({
	prompt,
	setPrompt,
	onSubmit,
	isLoading,
}) => {
	const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			onSubmit();
		}
	};

	return (
		<div className="relative">
			<textarea
				value={prompt}
				onChange={(e) => setPrompt(e.target.value)}
				onKeyDown={handleKeyDown}
				placeholder="Ask about a legal concept, case, or procedure..."
				className="w-full h-24 p-4 pr-16 bg-slate-800 border border-slate-700 rounded-xl focus:ring-2 focus:ring-cyan-500 focus:outline-none transition duration-200 resize-none text-white placeholder-slate-500"
				disabled={isLoading}
			/>
			<button
				onClick={onSubmit}
				disabled={isLoading || !prompt}
				className="absolute right-3 top-1/2 -translate-y-1/2 p-3 bg-cyan-600 text-white rounded-lg hover:bg-cyan-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-cyan-500"
			>
				{isLoading ? <LoadingSpinner /> : <PaperAirplaneIcon />}
			</button>
		</div>
	);
};

export default PromptInput;
