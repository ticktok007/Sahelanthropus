// src/App.tsx
import React, { useState } from 'react';
import { Landing } from './pages/Landing';
import { Workbench } from './pages/Workbench';
import { BenchmarkExample } from './types/superoptimizer';
import { BENCHMARK_EXAMPLES } from './services/mockService';

export function App() {
  const [view, setView] = useState<'landing' | 'workbench'>('workbench');
  const [selectedExample, setSelectedExample] = useState<BenchmarkExample>(BENCHMARK_EXAMPLES[4]);

  const handleSelectExample = (ex: BenchmarkExample) => {
    setSelectedExample(ex);
  };

  return (
    <div>
      {view === 'landing' ? (
        <Landing
          onOpenWorkbench={() => setView('workbench')}
          onSelectExample={handleSelectExample}
        />
      ) : (
        <Workbench
          initialCode={selectedExample.code}
          onGoToLanding={() => setView('landing')}
        />

      )}
    </div>
  );
}

export default App;
