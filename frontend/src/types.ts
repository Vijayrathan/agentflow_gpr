export interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  content: string;
  stageIndex: number;
  timestamp: Date;
}

export interface DatasetInfo {
  datasetName: string;
  numGenerated: number;
}

export interface SimulationResult {
  succeeded: number;
  failed: number;
  skipped: number;
  total: number;
  output_dir: string;
}

export interface WSIncoming {
  type: 'agent_message' | 'stage_change' | 'dataset_ready' | 'dataset_dismiss' | 'simulation_complete' | 'error';
  content?: string;
  stage_index?: number;
  stage_name?: string;
  dataset_name?: string;
  num_generated?: number;
  result?: SimulationResult;
  message?: string;
}

export const STAGE_NAMES = [
  'Layer Extraction',
  'Antenna & Waveform',
  'Model & Domain',
  'Advanced Parameters',
  'Dataset Generation',
  'Simulation',
];
