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

export interface WSIncoming {
  type: 'agent_message' | 'stage_change' | 'dataset_ready' | 'dataset_dismiss' | 'error';
  content?: string;
  stage_index?: number;
  stage_name?: string;
  dataset_name?: string;
  num_generated?: number;
  message?: string;
}

export const STAGE_NAMES = [
  'Layer Extraction',
  'Antenna & Waveform',
  'Model & Domain',
  'Advanced Parameters',
  'Dataset Generation',
];
