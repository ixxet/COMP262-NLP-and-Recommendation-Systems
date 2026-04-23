export type SentimentLabel = 'negative' | 'neutral' | 'positive';

export interface DatasetSummary {
  name: string;
  raw_rows: number;
  clean_rows: number;
  unique_users: number;
  unique_products: number;
  missing_review_text_raw: number;
  duplicate_clean_key_rows: number;
}

export interface ModelMetric {
  model: string;
  accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  weighted_f1: number;
}

export interface RecommenderMetric {
  model: string;
  mae: number;
  rmse: number;
}

export interface ProjectSummary {
  project: Record<string, string>;
  datasets: DatasetSummary[];
  phase1_sample: Record<string, number>;
  phase2_modeling: Record<string, number>;
  model_metrics: ModelMetric[];
  recommender_metrics: RecommenderMetric[];
  findings: string[];
}

export interface SentimentResponse {
  model: string;
  label: SentimentLabel;
  score: number;
  confidence: number;
  rating_label: SentimentLabel | null;
  rationale: string[];
}

export interface EnhanceResponse {
  actual_rating: number;
  sentiment_score: number;
  sentiment_virtual_rating: number;
  sentiment_weight: number;
  enhanced_rating: number;
  interpretation: string;
}

export interface SummaryExample {
  review_number: number;
  rating: number;
  label: SentimentLabel;
  original_word_count: number;
  summary_word_count: number;
  summary: string;
}

export interface LlmExamples {
  summarization_model: string;
  response_model: string;
  summaries: SummaryExample[];
  service_response: Record<string, string>;
}

export interface EvidenceHit {
  id: string;
  title: string;
  body: string;
  score: number;
}

export interface AskResponse {
  answer: string;
  evidence: EvidenceHit[];
}
