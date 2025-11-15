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
			setIsLoading(true);
			await new Promise((resolve) => setTimeout(resolve, 3000)); // Simulate network delay
			// What are the key considerations when drafting a non-compete clause for an employment contract in California?
			setResponse(
				`
# Key Considerations for Drafting a Non-Compete Clause in California

In California, non-compete clauses are generally unenforceable under Business and Professions Code §16600,
which prohibits contracts that restrain someone from engaging in a lawful profession, trade, or business.
However, there are limited exceptions and strategic considerations:

## 1. Statutory Exceptions:
    Sale of Business Exception: Permitted when a person sells the goodwill of a business (§16601).
    Dissolution of Partnership or LLC Exception: Allowed to protect the business’s goodwill during dissolution (§§16602, 16602.5).

## 2. Trade Secret Protection:
    While a non-compete is invalid, employers may still protect trade secrets under the California Uniform Trade Secrets Act (CUTSA).
    Clauses emphasizing confidentiality and non-solicitation of clients based on confidential information are often more defensible.

## 3. Narrow Drafting:
    Focus on non-solicitation and confidentiality rather than blanket prohibitions on employment.
    Overly broad restrictions risk being voided entirely.

## 4. Public Policy Trend:
    Recent California legislation (e.g., SB 699, effective 2024) expands penalties against employers
    who attempt to enforce void non-competes — even those signed outside California.

## 5. Practical Tip:
    Employers should instead use robust NDAs, IP assignment agreements,
    and limited-term garden leave provisions (if applicable and compliant).

## In short:
 A California non-compete will almost always be void unless tied to the sale of a business or similar transaction.
 Draft around it with narrowly tailored confidentiality and trade secret protections.


This is a placeholder response. The API integration is not yet implemented.`
			);
			setIsLoading(false);
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
				{response.length === 0 && <Header />}
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
