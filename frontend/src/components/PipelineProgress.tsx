import { STAGE_NAMES } from '../types';

interface Props {
  currentStage: number;
}

export default function PipelineProgress({ currentStage }: Props) {
  return (
    <div className="border-b border-gray-700 bg-gray-800/50 px-4 py-3">
      <div className="mx-auto max-w-5xl flex items-center justify-between">
        {STAGE_NAMES.map((name, i) => {
          const isComplete = i < currentStage;
          const isActive = i === currentStage;
          const isPending = i > currentStage;

          return (
            <div key={name} className="flex items-center">
              {i > 0 && (
                <div
                  className={`hidden sm:block h-0.5 w-6 lg:w-12 mx-1 ${
                    isComplete ? 'bg-emerald-500' : 'bg-gray-600'
                  }`}
                />
              )}
              <div className="flex flex-col items-center gap-1">
                <div
                  className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-colors ${
                    isComplete
                      ? 'bg-emerald-500 text-white'
                      : isActive
                        ? 'bg-blue-600 text-white ring-2 ring-blue-400'
                        : 'bg-gray-600 text-gray-400'
                  }`}
                >
                  {isComplete ? (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                      <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    i + 1
                  )}
                </div>
                <span
                  className={`text-[10px] leading-tight text-center max-w-16 lg:max-w-20 ${
                    isActive ? 'text-blue-400 font-medium' : isPending ? 'text-gray-500' : 'text-emerald-400'
                  }`}
                >
                  {name}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
