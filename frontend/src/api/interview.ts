import { apiClient, authenticatedFetch } from './client'

export interface InterviewMessage {
  id: number
  role: 'assistant' | 'user'
  content: string
  question_index: number
  is_followup: number
  followup_index: number
  created_at: string
}

export interface InterviewSession {
  id: number
  target_position: string
  difficulty: 'easy' | 'medium' | 'hard'
  status: 'preparing' | 'active' | 'finished'
  current_question_index: number
  current_dimension?: string | null
  current_plan_focus?: string | null
  total_question_count: number
  created_at: string
  updated_at: string
  messages: InterviewMessage[]
}

export interface InterviewScore {
  id: number
  session_id: number
  question_index: number
  question: string
  answer: string
  dimension: string
  score: number
  sub_scores: Record<string, number>
  reason: string
  weaknesses: string[]
  suggestions: string[]
  is_fallback: boolean
  fallback_reason?: string | null
  created_at: string
}

export interface InterviewCreateInput {
  target_position: string
  difficulty: 'easy' | 'medium' | 'hard'
  resume_id?: number
  job_description_id?: number
}

const sessionCache = new Map<number, InterviewSession>()

export function rememberInterview(session: InterviewSession) {
  sessionCache.set(session.id, session)
}

export function takeRememberedInterview(id: number) {
  const session = sessionCache.get(id) || null
  sessionCache.delete(id)
  return session
}

export function warmupInterview(targetPosition: string) {
  return apiClient.post<void>('/interviews/warmup', { target_position: targetPosition })
}

export function createInterview(data: InterviewCreateInput) {
  return apiClient.post<InterviewSession>('/interviews', data)
}

export function fetchInterviews() {
  return apiClient.get<InterviewSession[]>('/interviews')
}

export function fetchInterview(id: number) {
  return apiClient.get<InterviewSession>(`/interviews/${id}`)
}

export function fetchInterviewScores(id: number) {
  return apiClient.get<InterviewScore[]>(`/interviews/${id}/scores`)
}

export function startInterview(id: number) {
  return apiClient.post<InterviewSession>(`/interviews/${id}/start`)
}

export function answerInterview(id: number, answer: string, requestId: string) {
  return apiClient.post<InterviewSession>(`/interviews/${id}/answer`, {
    answer,
    request_id: requestId
  })
}

export function finishInterview(id: number) {
  return apiClient.post<InterviewSession>(`/interviews/${id}/finish`)
}

export type InterviewStreamEvent =
  | { event: 'status'; data: { phase: string } }
  | { event: 'delta'; data: { content: string } }
  | { event: 'text_done'; data: { content: string } }
  | { event: 'complete'; data: InterviewSession }
  | { event: 'error'; data: { message: string } }

export function streamInterviewStart(
  id: number,
  onEvent: (event: InterviewStreamEvent) => void,
  signal?: AbortSignal
) {
  return consumeInterviewStream(`/api/interviews/${id}/start/stream`, undefined, onEvent, signal)
}
export async function streamAnswer(
  id: number,
  answer: string,
  requestId: string,
  onEvent: (event: InterviewStreamEvent) => void,
  signal?: AbortSignal
) {
  return consumeInterviewStream(
    `/api/interviews/${id}/answer/stream`,
    { answer, request_id: requestId },
    onEvent,
    signal
  )
}

async function consumeInterviewStream(
  url: string,
  body: object | undefined,
  onEvent: (event: InterviewStreamEvent) => void,
  signal?: AbortSignal
) {
  // 统一消费启动和回答接口的 SSE；complete 事件才是后端已提交的最终会话快照。
  const response = await authenticatedFetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream'
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
    signal
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const payload = await response.json()
      if (typeof payload.detail === 'string') message = payload.detail
    } catch {
      // Keep the status-based message for non-JSON responses.
    }
    throw new Error(message)
  }
  if (!response.body) throw new Error('Streaming response body is unavailable')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, '\n')
    let boundary = buffer.indexOf('\n\n')
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const parsed = parseSseFrame(frame)
      if (parsed?.event === 'error') throw new Error(parsed.data.message)
      if (parsed) onEvent(parsed)
      boundary = buffer.indexOf('\n\n')
    }
    if (done) break
  }
}

function parseSseFrame(frame: string): InterviewStreamEvent | null {
  // 支持注释心跳与多行 data，避免网络分片边界影响上层业务事件。
  let event = 'message'
  const dataLines: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith(':')) continue
    if (line.startsWith('event:')) event = line.slice(6).trim()
    if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
  }
  if (dataLines.length === 0) return null
  return { event, data: JSON.parse(dataLines.join('\n')) } as InterviewStreamEvent
}
