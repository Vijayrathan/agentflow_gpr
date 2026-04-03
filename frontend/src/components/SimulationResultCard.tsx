import type { SimulationResult } from '../types';

interface Props {
  result: SimulationResult;
}

export default function SimulationResultCard({ result }: Props) {
  const allPassed = result.failed === 0;
  const borderColor = allPassed ? 'border-emerald-700' : 'border-amber-700';
  const bgColor = allPassed ? 'bg-emerald-900/30' : 'bg-amber-900/30';
  const titleColor = allPassed ? 'text-emerald-300' : 'text-amber-300';
  const iconColor = allPassed ? 'text-emerald-400' : 'text-amber-400';

  return (
    <div className="border-t border-gray-700 bg-gray-800 p-4">
      <div className="mx-auto max-w-3xl">
        <div className={`rounded-xl border ${borderColor} ${bgColor} p-4`}>
          <div className="flex items-center gap-2 mb-3">
            {allPassed ? (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={`h-5 w-5 ${iconColor}`}>
                <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={`h-5 w-5 ${iconColor}`}>
                <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495ZM10 5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 10 5Zm0 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z" clipRule="evenodd" />
              </svg>
            )}
            <h3 className={`text-sm font-semibold ${titleColor}`}>
              Simulation Complete
            </h3>
          </div>
          <div className="grid grid-cols-4 gap-3 text-center text-sm">
            <div>
              <p className="text-gray-400">Succeeded</p>
              <p className="text-lg font-bold text-emerald-400">{result.succeeded}</p>
            </div>
            <div>
              <p className="text-gray-400">Failed</p>
              <p className={`text-lg font-bold ${result.failed > 0 ? 'text-red-400' : 'text-gray-500'}`}>{result.failed}</p>
            </div>
            <div>
              <p className="text-gray-400">Skipped</p>
              <p className="text-lg font-bold text-gray-400">{result.skipped}</p>
            </div>
            <div>
              <p className="text-gray-400">Total</p>
              <p className="text-lg font-bold text-white">{result.total}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}