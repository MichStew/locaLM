import React from 'react';

const Header: React.FC = () => {
	return (
		<header className="text-center py-8 sm:py-12">
			<h1
				className="text-6xl sm:text-7xl md:text-8xl font-black text-slate-300 tracking-wider"
				style={{
					textShadow: `
            0 1px 0 #bbb,
            0 2px 0 #aaa,
            0 3px 0 #999,
            0 4px 1px rgba(0,0,0,0.4),
            0 0 5px rgba(0,0,0,0.4),
            0 1px 3px rgba(0,0,0,0.6),
            0 3px 5px rgba(0,0,0,0.4),
            0 5px 10px rgba(0,0,0,0.35),
            0 10px 10px rgba(0,0,0,0.3),
            0 20px 20px rgba(0,0,0,0.25)
          `,
				}}
			>
				LocaLM
			</h1>
			<p className="text-slate-400 mt-2 text-lg">
				Your AI-Powered Legal Insight Partner
			</p>
		</header>
	);
};

export default Header;
