import type { AskResponse, EnhanceResponse, LlmExamples, ProjectSummary, SentimentResponse } from './types';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) },
    ...init
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export function getSummary() {
  return request<ProjectSummary>('/v1/summary');
}

export function getExamples() {
  return request<LlmExamples>('/v1/examples/summaries');
}

export function predictSentiment(text: string, rating?: number) {
  return request<SentimentResponse>('/v1/sentiment/predict', {
    method: 'POST',
    body: JSON.stringify({ text, rating })
  });
}

export function enhanceRating(actual_rating: number, sentiment_score: number, sentiment_weight: number) {
  return request<EnhanceResponse>('/v1/recommend/enhance', {
    method: 'POST',
    body: JSON.stringify({ actual_rating, sentiment_score, sentiment_weight })
  });
}

export function askProject(question: string) {
  return request<AskResponse>('/v1/assistant/ask', {
    method: 'POST',
    body: JSON.stringify({ question, max_evidence: 3 })
  });
}
