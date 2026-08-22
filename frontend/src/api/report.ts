import { apiClient } from './client'

export interface InterviewReport {
  id: number
  session_id: number
  total_score: number
  summary: string
  strengths: string[]
  weaknesses: string[]
  suggestions: string[]
  dimension_scores: ReportDimensionScore[]
  learning_path: string[]
  sample_answer: string
  citations: KnowledgeCitation[]
  created_at: string
  updated_at: string
}

export interface KnowledgeContextChunkCitation {
  chunk_id: string
  chunk_index: number
  source_page: number | null
  is_primary: boolean
  context_token_count: number
  context_truncated: boolean
}

export interface KnowledgeCitation {
  reference: number
  query: string
  purpose: string | null
  question_index: number | null
  document_id: number
  title: string
  category: string
  target_position: string
  index_version: number
  chunk_id: string
  chunk_index: number
  source_page: number | null
  section_title: string | null
  retrieval_score: number
  retrieval_routes: string[]
  retrieval_scores: Record<string, number>
  retrieval_ranks: Record<string, number>
  rrf_score: number
  rerank_score: number
  rerank_provider: string | null
  rerank_model: string | null
  rerank_rank: number | null
  rerank_is_fallback: boolean
  rerank_fallback_reason: string | null
  context_chunks: KnowledgeContextChunkCitation[]
  context_token_count: number
  context_truncated: boolean
}

export interface ReportDimensionScore {
  dimension: string
  score: number
  question_indexes: number[]
  focus: string
  weaknesses: string[]
  suggestions: string[]
}

export interface InterviewReportListItem {
  id: number
  session_id: number
  target_position: string
  difficulty: string
  total_score: number
  created_at: string
  updated_at: string
}

export function fetchReports() {
  return apiClient.get<InterviewReportListItem[]>('/reports')
}

export function fetchReport(id: number) {
  return apiClient.get<InterviewReport>(`/reports/${id}`)
}

export function fetchInterviewReport(sessionId: number) {
  return apiClient.get<InterviewReport>(`/interviews/${sessionId}/report`)
}
