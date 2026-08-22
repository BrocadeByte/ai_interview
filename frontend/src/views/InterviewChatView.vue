<script setup lang="ts">
import { ElButton, ElInput, ElMessage, ElProgress, ElResult, ElSkeleton, ElTag } from 'element-plus'
import 'element-plus/theme-chalk/el-button.css'
import 'element-plus/theme-chalk/el-input.css'
import 'element-plus/theme-chalk/el-message.css'
import 'element-plus/theme-chalk/el-progress.css'
import 'element-plus/theme-chalk/el-result.css'
import 'element-plus/theme-chalk/el-skeleton.css'
import 'element-plus/theme-chalk/el-tag.css'
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { fetchInterview, fetchInterviewScores, finishInterview, streamAnswer, streamInterviewStart, takeRememberedInterview, type InterviewMessage, type InterviewScore, type InterviewSession } from '../api/interview'
import { getApiErrorMessage } from '../api/client'
import ChatMessage from '../components/interview/ChatMessage.vue'
import LiveScorePanel from '../components/interview/LiveScorePanel.vue'
import { createRequestId } from '../utils/request-id'

const route = useRoute()
const router = useRouter()
const session = ref<InterviewSession | null>(null)
const answer = ref('')
const loading = ref(false)
const finishing = ref(false)
const starting = ref(false)
const streamingAssistantMessage = ref<InterviewMessage | null>(null)
const streamPhase = ref('')
const streamTextDone = ref(false)
const pendingUserMessage = ref<InterviewMessage | null>(null)
const chatBox = ref<HTMLElement | null>(null)
const interviewPage = ref<HTMLElement | null>(null)
const retryableAnswer = ref<{ answer: string; requestId: string } | null>(null)
const initialLoading = ref(true)
const scores = ref<InterviewScore[]>([])
const scoresLoading = ref(false)
const scoresError = ref(false)
let scoreRefreshQueued = false

const sessionId = Number(route.params.id)
const latestScore = computed(() => scores.value[scores.value.length - 1] || null)
const mobileScoreLabel = computed(() => latestScore.value
  ? `实时评分，已完成 ${scores.value.length} 轮评分，最新 ${latestScore.value.score} 分`
  : '实时评分，完成作答后查看反馈')

const lastAssistantMessageId = computed(() => {
  const messages = session.value?.messages || []
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'assistant') return messages[index].id
  }
  return null
})

const progressPercentage = computed(() => {
  if (!session.value) return 0
  const totalQuestionCount = session.value.total_question_count || 8
  const current = Math.min(session.value.current_question_index, totalQuestionCount)
  return Math.round((current / totalQuestionCount) * 100)
})

const latestAssistantMessage = computed(() => {
  const messages = session.value?.messages || []
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'assistant') return messages[index]
  }
  return null
})

const visibleAssistantMessage = computed(() => {
  if (streamingAssistantMessage.value?.content) return streamingAssistantMessage.value
  return latestAssistantMessage.value
})

const latestQuestionLabel = computed(() => {
  const message = visibleAssistantMessage.value
  if (!message) return '准备中'
  if (message.is_followup) return `第 ${message.question_index} 题追问 ${message.followup_index}`
  return `第 ${message.question_index} 题主问题`
})

const isPreparing = computed(() => {
  if (!session.value) return false
  return session.value.status === 'preparing' || (session.value.status === 'active' && !latestAssistantMessage.value)
})

const hasVisibleQuestion = computed(() => Boolean(visibleAssistantMessage.value?.content))
const isPreparingWithVisibleQuestion = computed(() => isPreparing.value && hasVisibleQuestion.value)
const isFinalizingFirstQuestion = computed(() => isPreparingWithVisibleQuestion.value && streamTextDone.value)
const canAnswer = computed(() => session.value?.status === 'active' && Boolean(latestAssistantMessage.value))

const headerStatusLabel = computed(() => {
  if (isFinalizingFirstQuestion.value) return '首题已生成'
  if (isPreparingWithVisibleQuestion.value) return '首题生成中'
  if (isPreparing.value) return '准备中'
  if (session.value?.status === 'active') return '面试进行中'
  return '面试已结束'
})

const preparationMessage = computed(() => {
  if (streamTextDone.value) return '首题已生成，正在整理本场面试计划...'
  if (streamingAssistantMessage.value?.content) return '第一道问题正在生成...'
  if (streamPhase.value === 'retrieving') return '正在连接知识库并匹配岗位重点...'
  return '正在准备面试环境...'
})

async function scrollToBottom() {
  await nextTick()
  if (chatBox.value) {
    chatBox.value.scrollTop = chatBox.value.scrollHeight
  }
}

async function loadScores() {
  if (scoresLoading.value) {
    scoreRefreshQueued = true
    return
  }
  scoresLoading.value = true
  scoresError.value = false
  try {
    scores.value = (await fetchInterviewScores(sessionId)).data
  } catch {
    scoresError.value = true
  } finally {
    scoresLoading.value = false
    if (scoreRefreshQueued) {
      scoreRefreshQueued = false
      void loadScores()
    }
  }
}

async function load() {
  try {
    const remembered = takeRememberedInterview(sessionId)
    const data = remembered || (await fetchInterview(sessionId)).data
    session.value = data
    void loadScores()
    await scrollToBottom()
    if (data.status === 'preparing' || (data.status === 'active' && data.messages.length === 0)) {
      await startFirstQuestion()
    }
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error, '加载面试失败'))
  } finally {
    initialLoading.value = false
  }
}

function syncVisualViewport() {
  const height = window.visualViewport?.height || window.innerHeight
  interviewPage.value?.style.setProperty('--interview-viewport-height', `${Math.round(height)}px`)
}

onMounted(() => {
  syncVisualViewport()
  window.visualViewport?.addEventListener('resize', syncVisualViewport)
  window.visualViewport?.addEventListener('scroll', syncVisualViewport)
  void load()
})

onBeforeUnmount(() => {
  window.visualViewport?.removeEventListener('resize', syncVisualViewport)
  window.visualViewport?.removeEventListener('scroll', syncVisualViewport)
})

async function startFirstQuestion() {
  if (starting.value) return
  starting.value = true
  streamTextDone.value = false
  streamingAssistantMessage.value = {
    id: -Date.now(),
    role: 'assistant',
    content: '',
    question_index: 1,
    is_followup: 0,
    followup_index: 0,
    created_at: new Date().toISOString()
  }
  try {
    await streamInterviewStart(sessionId, (event) => {
      if (event.event === 'status') {
        streamPhase.value = event.data.phase
      } else if (event.event === 'delta' && streamingAssistantMessage.value) {
        streamingAssistantMessage.value.content += event.data.content
        void scrollToBottom()
      } else if (event.event === 'text_done') {
        if (streamingAssistantMessage.value) streamingAssistantMessage.value.content = event.data.content
        streamTextDone.value = true
      } else if (event.event === 'complete') {
        session.value = event.data
        streamingAssistantMessage.value = null
      }
    })
    await scrollToBottom()
  } catch (error) {
    streamingAssistantMessage.value = null
    const message = error instanceof Error ? error.message : getApiErrorMessage(error, 'Failed to generate the first question')
    ElMessage.error(message)
  } finally {
    starting.value = false
    streamPhase.value = ''
    streamTextDone.value = false
  }
}
async function submitAnswer() {
  const submittedAnswer = answer.value.trim()
  if (!submittedAnswer || loading.value) return

  loading.value = true
  streamTextDone.value = false
  let requestId = retryableAnswer.value?.answer === submittedAnswer
    ? retryableAnswer.value.requestId
    : ''

  try {
    requestId ||= createRequestId()
    pendingUserMessage.value = {
      id: -Date.now(),
      role: 'user',
      content: submittedAnswer,
      question_index: session.value?.current_question_index || 1,
      is_followup: latestAssistantMessage.value?.is_followup || 0,
      followup_index: latestAssistantMessage.value?.followup_index || 0,
      created_at: new Date().toISOString()
    }
    streamingAssistantMessage.value = {
      id: -(Date.now() + 1),
      role: 'assistant',
      content: '',
      question_index: session.value?.current_question_index || 1,
      is_followup: 0,
      followup_index: 0,
      created_at: new Date().toISOString()
    }
    answer.value = ''
    await scrollToBottom()

    await streamAnswer(sessionId, submittedAnswer, requestId, (event) => {
      if (event.event === 'status') {
        streamPhase.value = event.data.phase
      } else if (event.event === 'delta' && streamingAssistantMessage.value) {
        streamingAssistantMessage.value.content += event.data.content
        void scrollToBottom()
      } else if (event.event === 'text_done') {
        if (streamingAssistantMessage.value) streamingAssistantMessage.value.content = event.data.content
        streamTextDone.value = true
      } else if (event.event === 'complete') {
        session.value = event.data
        pendingUserMessage.value = null
        streamingAssistantMessage.value = null
        retryableAnswer.value = null
        void loadScores()
      } else if (event.event === 'error') {
        throw new Error(event.data.message)
      }
    })
    await scrollToBottom()
  } catch (error) {
    answer.value = submittedAnswer
    if (requestId) retryableAnswer.value = { answer: submittedAnswer, requestId }
    pendingUserMessage.value = null
    streamingAssistantMessage.value = null
    const message = error instanceof Error ? error.message : getApiErrorMessage(error, 'Submit failed')
    ElMessage.error(message)
  } finally {
    loading.value = false
    streamPhase.value = ''
    streamTextDone.value = false
  }
}
async function finish() {
  if (finishing.value) return
  finishing.value = true
  try {
    const { data } = await finishInterview(sessionId)
    session.value = data
    await scrollToBottom()
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error, '结束面试失败'))
  } finally {
    finishing.value = false
  }
}
</script>

<template>
  <main ref="interviewPage" class="app-page interview-page">
    <nav class="topbar">
      <strong>AI 模拟面试</strong>
      <div>
        <el-button text @click="router.push('/reports')">报告</el-button>
        <el-button text @click="router.push('/interviews')">返回</el-button>
      </div>
    </nav>

    <section v-if="session" class="content chat-layout">
      <header class="chat-header">
        <div>
          <h1>{{ session.target_position }}</h1>
          <p>{{ headerStatusLabel }} · {{ latestQuestionLabel }}</p>
        </div>
        <div class="chat-actions">
          <el-button v-if="canAnswer" :loading="finishing" @click="finish">结束面试</el-button>
          <el-button v-else-if="isPreparing" :loading="starting && !streamTextDone" disabled>
            {{ isFinalizingFirstQuestion ? '完成计划中' : '生成首题中' }}
          </el-button>
          <el-button v-else type="primary" @click="router.push(`/interviews/${session.id}/report`)">查看报告</el-button>
        </div>
      </header>

      <section class="interview-status-panel">
        <div class="status-metric">
          <span>进度</span>
          <strong>{{ Math.min(session.current_question_index, session.total_question_count || 8) }} / {{ session.total_question_count || 8 }}</strong>
        </div>
        <el-progress :percentage="progressPercentage" :stroke-width="10" />
        <el-tag :type="isPreparing && !hasVisibleQuestion ? 'info' : visibleAssistantMessage?.is_followup ? 'warning' : 'success'" effect="light">
          {{ isPreparing && !hasVisibleQuestion ? '准备中' : visibleAssistantMessage?.is_followup ? '追问中' : '主问题' }}
        </el-tag>
        <div class="status-focus">
          <span>当前维度</span>
          <strong>{{ isPreparing ? (streamTextDone ? '正在完成面试计划' : '生成面试计划中') : session.current_dimension || '未指定' }}</strong>
          <small>{{ isPreparing ? (streamTextDone ? '首题已生成，计划完成后即可作答。' : 'AI 正在生成第一道问题，请稍候。') : session.current_plan_focus || '暂无考察重点' }}</small>
        </div>
      </section>

      <details class="mobile-score-disclosure">
        <summary :aria-label="mobileScoreLabel" aria-controls="mobile-live-score-panel">
          <span>
            <strong>实时评分</strong>
            <small>{{ latestScore ? `已完成 ${scores.length} 轮评分` : '完成作答后查看反馈' }}</small>
          </span>
          <b>{{ latestScore ? `${latestScore.score} 分` : '查看' }}</b>
        </summary>
        <LiveScorePanel
          id="mobile-live-score-panel"
          class="mobile-live-score"
          :scores="scores"
          :loading="scoresLoading"
          :error="scoresError"
          @retry="loadScores"
        />
      </details>

      <div class="interview-workspace">
        <div ref="chatBox" class="chat-box" role="log" aria-label="面试问答记录" aria-live="polite" aria-relevant="additions text" tabindex="0">
          <ChatMessage
            v-for="message in session.messages"
            :key="message.id"
            :role="message.role"
            :content="message.content"

          />
          <ChatMessage
            v-if="pendingUserMessage"
            :key="pendingUserMessage.id"
            :role="pendingUserMessage.role"
            :content="pendingUserMessage.content"
          />
          <ChatMessage
            v-if="streamingAssistantMessage?.content"
            :key="streamingAssistantMessage.id"
            :role="streamingAssistantMessage.role"
            :content="streamingAssistantMessage.content"
            :streaming="!streamTextDone"
          />
          <div v-if="(starting || isPreparing) && !streamingAssistantMessage?.content" class="thinking-indicator" aria-live="polite"><span class="thinking-pulse" aria-hidden="true"></span>{{ preparationMessage }}</div>
          <div v-else-if="loading && !streamingAssistantMessage?.content" class="thinking-indicator" role="status" aria-live="polite">AI 正在评分并生成下一步问题...</div>
        </div>

        <LiveScorePanel
          class="desktop-live-score"
          :scores="scores"
          :loading="scoresLoading"
          :error="scoresError"
          @retry="loadScores"
        />
      </div>

      <form v-if="canAnswer" class="answer-box answer-composer" @submit.prevent="submitAnswer">
        <el-input v-model="answer" type="textarea" :autosize="{ minRows: 2, maxRows: 5 }" aria-label="本轮回答" placeholder="输入回答，建议说明背景、行动和结果" :disabled="loading" />
        <el-button type="primary" native-type="submit" :loading="loading && !streamTextDone" :disabled="loading">{{ loading && streamTextDone ? '正在保存本轮结果…' : '提交回答' }}</el-button>
      </form>

      <div v-else-if="isPreparingWithVisibleQuestion" class="answer-box answer-composer">
        <el-input type="textarea" :autosize="{ minRows: 2, maxRows: 5 }" aria-label="本轮回答" :placeholder="streamTextDone ? '首题已生成，面试计划完成后即可作答' : '首题正在生成，请稍候'" disabled />
        <el-button type="primary" loading disabled>{{ streamTextDone ? '完成计划中' : '生成首题中' }}</el-button>
      </div>

      <el-result v-else-if="isPreparing" icon="info" title="正在准备面试" sub-title="AI 正在生成第一道问题，请稍候。">
        <template #extra>
          <el-button type="primary" :loading="starting" @click="startFirstQuestion">重新生成</el-button>
        </template>
      </el-result>

      <el-result v-else icon="success" title="面试完成" sub-title="问答记录已保存，可以查看评分和复盘报告。">
        <template #extra>
          <el-button type="primary" @click="router.push(`/interviews/${session.id}/report`)">查看报告</el-button>
          <el-button @click="router.push('/reports')">历史报告</el-button>
        </template>
      </el-result>
    </section>

    <section v-else-if="initialLoading" class="content interview-loading" aria-live="polite">
      <el-skeleton :rows="6" animated />
    </section>
  </main>
</template>
