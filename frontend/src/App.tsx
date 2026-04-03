import { useMemo } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useChat } from './hooks/useChat';
import PipelineProgress from './components/PipelineProgress';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import FileDownloadCard from './components/FileDownloadCard';
import SimulationResultCard from './components/SimulationResultCard';

function App() {
  const sessionId = useMemo(() => uuidv4(), []);
  const {
    messages,
    currentStage,
    isTyping,
    isConnected,
    datasetInfo,
    simulationResult,
    error,
    sendMessage,
  } = useChat(sessionId);

  return (
    <div className="flex h-screen flex-col bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-700 bg-gray-800 px-4 py-3">
        <div className="mx-auto max-w-4xl flex items-center justify-between">
          <h1 className="text-lg font-semibold text-white">GPR Simulation Assistant</h1>
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400'}`}
            />
            <span className="text-xs text-gray-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Pipeline progress */}
      <PipelineProgress currentStage={currentStage} />

      {/* Error banner */}
      {error && (
        <div className="bg-red-900/50 border-b border-red-700 px-4 py-2">
          <p className="mx-auto max-w-3xl text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Chat area */}
      <ChatWindow messages={messages} isTyping={isTyping} />

      {/* Dataset download card */}
      {datasetInfo && <FileDownloadCard datasetInfo={datasetInfo} />}

      {/* Simulation result card */}
      {simulationResult && <SimulationResultCard result={simulationResult} />}

      {/* Input */}
      <ChatInput onSend={sendMessage} disabled={isTyping || !isConnected} />
    </div>
  );
}

export default App;
