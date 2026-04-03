import { useCallback, useEffect, useReducer, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type { ChatMessage, DatasetInfo, SimulationResult, WSIncoming } from '../types';

interface ChatState {
  messages: ChatMessage[];
  currentStage: number;
  stageName: string;
  isTyping: boolean;
  isConnected: boolean;
  datasetInfo: DatasetInfo | null;
  simulationResult: SimulationResult | null;
  error: string | null;
}

type ChatAction =
  | { type: 'connected' }
  | { type: 'disconnected' }
  | { type: 'user_message'; content: string; stageIndex: number }
  | { type: 'agent_message'; content: string; stageIndex: number }
  | { type: 'stage_change'; stageIndex: number; stageName: string }
  | { type: 'dataset_ready'; info: DatasetInfo }
  | { type: 'dataset_dismiss' }
  | { type: 'simulation_complete'; result: SimulationResult }
  | { type: 'error'; message: string }
  | { type: 'set_typing'; value: boolean };

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'connected':
      return { ...state, isConnected: true, error: null };
    case 'disconnected':
      return { ...state, isConnected: false };
    case 'user_message':
      return {
        ...state,
        isTyping: true,
        messages: [
          ...state.messages,
          {
            id: uuidv4(),
            role: 'user',
            content: action.content,
            stageIndex: action.stageIndex,
            timestamp: new Date(),
          },
        ],
      };
    case 'agent_message':
      return {
        ...state,
        isTyping: false,
        messages: [
          ...state.messages,
          {
            id: uuidv4(),
            role: 'agent',
            content: action.content,
            stageIndex: action.stageIndex,
            timestamp: new Date(),
          },
        ],
      };
    case 'stage_change':
      return {
        ...state,
        currentStage: action.stageIndex,
        stageName: action.stageName,
      };
    case 'dataset_ready':
      return { ...state, datasetInfo: action.info };
    case 'dataset_dismiss':
      return { ...state, datasetInfo: null };
    case 'simulation_complete':
      return { ...state, simulationResult: action.result, isTyping: false };
    case 'error':
      return { ...state, error: action.message, isTyping: false };
    case 'set_typing':
      return { ...state, isTyping: action.value };
    default:
      return state;
  }
}

const initialState: ChatState = {
  messages: [],
  currentStage: 0,
  stageName: 'Layer Extraction',
  isTyping: true,
  isConnected: false,
  datasetInfo: null,
  simulationResult: null,
  error: null,
};

export function useChat(sessionId: string) {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const stageRef = useRef(state.currentStage);
  stageRef.current = state.currentStage;

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => dispatch({ type: 'connected' });

    ws.onmessage = (event) => {
      const data: WSIncoming = JSON.parse(event.data);

      switch (data.type) {
        case 'agent_message':
          dispatch({
            type: 'agent_message',
            content: data.content ?? '',
            stageIndex: stageRef.current,
          });
          break;
        case 'stage_change':
          dispatch({
            type: 'stage_change',
            stageIndex: data.stage_index ?? 0,
            stageName: data.stage_name ?? '',
          });
          // Show typing indicator when entering the simulation stage
          if (data.stage_name === 'Simulation') {
            dispatch({ type: 'set_typing', value: true });
          }
          break;
        case 'dataset_ready':
          dispatch({
            type: 'dataset_ready',
            info: {
              datasetName: data.dataset_name ?? '',
              numGenerated: data.num_generated ?? 0,
            },
          });
          break;
        case 'dataset_dismiss':
          dispatch({ type: 'dataset_dismiss' });
          break;
        case 'simulation_complete':
          dispatch({
            type: 'simulation_complete',
            result: data.result!,
          });
          break;
        case 'error':
          dispatch({ type: 'error', message: data.message ?? 'Unknown error' });
          break;
      }
    };

    ws.onclose = () => dispatch({ type: 'disconnected' });
    ws.onerror = () => dispatch({ type: 'error', message: 'WebSocket connection failed' });

    return () => {
      ws.close();
    };
  }, [sessionId]);

  const sendMessage = useCallback(
    (text: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      dispatch({ type: 'user_message', content: text, stageIndex: stageRef.current });
      wsRef.current.send(JSON.stringify({ type: 'user_message', content: text }));
    },
    [],
  );

  return {
    messages: state.messages,
    currentStage: state.currentStage,
    stageName: state.stageName,
    isTyping: state.isTyping,
    isConnected: state.isConnected,
    datasetInfo: state.datasetInfo,
    simulationResult: state.simulationResult,
    error: state.error,
    sendMessage,
  };
}
