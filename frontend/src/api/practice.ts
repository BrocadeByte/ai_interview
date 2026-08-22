import { apiClient } from './client'

export type QuestionPracticeMode = 'repeat_question' | 'similar_question'

export interface PracticeCreationResponse {
  id: number
  status: 'not_started' | 'practicing' | 'ready_for_retest' | 'completed'
  practice_session_id?: number | null
}

export interface QuestionPracticeInput {
  report_id: number
  question_review_id: number
  score_id: number
  question_index: number
  weakness_key: string
  weakness_title: string
  practice_mode: QuestionPracticeMode
}

export interface ReportPracticeInput {
  report_id: number
  weakness_key: string
  weakness_title: string
}

export function createPracticeFromQuestionReview(data: QuestionPracticeInput) {
  return apiClient.post<PracticeCreationResponse>('/practice/from-question-review', data)
}

export function createPracticeFromReport(data: ReportPracticeInput) {
  return apiClient.post<PracticeCreationResponse>('/practice/from-report', data)
}
