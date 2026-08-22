<script setup lang="ts">
import { ArrowRight, Clock, MagicStick, Position } from '@element-plus/icons-vue'
import { ElButton, ElEmpty, ElForm, ElFormItem, ElIcon, ElInput, ElMessage, ElSegmented, ElTag } from 'element-plus'
import 'element-plus/theme-chalk/el-button.css'
import 'element-plus/theme-chalk/el-empty.css'
import 'element-plus/theme-chalk/el-form.css'
import 'element-plus/theme-chalk/el-icon.css'
import 'element-plus/theme-chalk/el-input.css'
import 'element-plus/theme-chalk/el-message.css'
import 'element-plus/theme-chalk/el-segmented.css'
import 'element-plus/theme-chalk/el-tag.css'
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'

import { getApiErrorMessage } from '../api/client'
import { createInterview, fetchInterviews, rememberInterview, type InterviewSession, warmupInterview } from '../api/interview'
import { fetchProfile } from '../api/profile'

const router = useRouter()
const loading = ref(false)
const sessions = ref<InterviewSession[]>([])
const form = reactive({ target_position: '', difficulty: 'medium' })
let warmupTimer: ReturnType<typeof setTimeout> | null = null
let lastWarmedPosition = ''

const finishedCount = computed(() => sessions.value.filter((session) => session.status === 'finished').length)
const activeCount = computed(() => sessions.value.filter((session) => session.status !== 'finished').length)

async function load() {
  const [{ data: profile }, { data: history }] = await Promise.all([fetchProfile(), fetchInterviews()])
  form.target_position = profile.target_position || ''
  sessions.value = history
}

function scheduleWarmup(position: string) {
  const normalized = position.trim()
  if (warmupTimer) clearTimeout(warmupTimer)
  if (!normalized || normalized === lastWarmedPosition) return
  warmupTimer = setTimeout(() => {
    lastWarmedPosition = normalized
    void warmupInterview(normalized).catch(() => undefined)
  }, 500)
}

watch(() => form.target_position, scheduleWarmup)
onMounted(load)
onBeforeUnmount(() => {
  if (warmupTimer) clearTimeout(warmupTimer)
})

async function startInterview() {
  if (!form.target_position.trim()) {
    ElMessage.warning('请先填写目标岗位')
    return
  }
  loading.value = true
  try {
    const normalizedPosition = form.target_position.trim()
    void import('./InterviewChatView.vue')
    void warmupInterview(normalizedPosition).catch(() => undefined)
    const { data } = await createInterview({ ...form, target_position: normalizedPosition })
    rememberInterview(data)
    await router.push(`/interviews/${data.id}`)
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error, '创建面试失败'))
  } finally {
    loading.value = false
  }
}

function statusLabel(status: InterviewSession['status']) {
  return status === 'preparing' ? '准备中' : status === 'active' ? '进行中' : '已完成'
}

function statusType(status: InterviewSession['status']) {
  return status === 'active' ? 'primary' : status === 'finished' ? 'success' : 'info'
}

function difficultyLabel(value: InterviewSession['difficulty']) {
  return { easy: '基础', medium: '标准', hard: '进阶' }[value]
}
</script>

<template>
  <main class="app-page home-page">
    <section class="content dashboard-page">
      <header class="page-heading home-heading">
        <div>
          <span class="eyebrow">JOB-SPECIFIC INTERVIEW TRAINING</span>
          <h1>开始岗位定制训练</h1>
          <p>根据目标技术岗位和难度生成专属训练流程，完成后获得多维复盘报告。</p>
        </div>
        <div class="overview-metrics" aria-label="训练概览">
          <div><span>累计训练</span><strong>{{ sessions.length }}</strong></div>
          <div><span>已完成</span><strong>{{ finishedCount }}</strong></div>
          <div><span>进行中</span><strong>{{ activeCount }}</strong></div>
        </div>
      </header>

      <div class="dashboard">
        <section class="create-panel primary-work-panel">
          <div class="panel-title-row">
            <span class="section-icon"><el-icon><MagicStick /></el-icon></span>
            <div>
              <h2>训练设置</h2>
              <p>选择你的目标技术岗位与训练强度</p>
            </div>
          </div>
          <el-form label-position="top" class="interview-form" @submit.prevent="startInterview">
            <el-form-item label="目标岗位">
              <el-input v-model="form.target_position" placeholder="例如：前端开发工程师" :prefix-icon="Position" size="large" />
            </el-form-item>
            <el-form-item label="面试难度">
              <el-segmented
                v-model="form.difficulty"
                :options="[{ label: '基础', value: 'easy' }, { label: '标准', value: 'medium' }, { label: '进阶', value: 'hard' }]"
                size="large"
              />
            </el-form-item>
            <div class="difficulty-hint">
              <el-icon><Clock /></el-icon>
              <span>预计 20–30 分钟，共约 8 轮核心问答</span>
            </div>
            <el-button type="primary" native-type="submit" :loading="loading" size="large" class="start-button">
              开始训练<el-icon class="el-icon--right"><ArrowRight /></el-icon>
            </el-button>
          </el-form>
        </section>

        <aside class="history-panel">
          <div class="panel-heading">
            <div><h2>最近训练</h2><p>继续训练或查看历史复盘</p></div>
            <el-button text @click="router.push('/reports')">全部报告</el-button>
          </div>
          <el-empty v-if="sessions.length === 0" description="暂无训练记录" :image-size="72" />
          <div v-else class="session-list">
            <button v-for="session in sessions.slice(0, 6)" :key="session.id" type="button" class="session-item" :aria-label="`${statusLabel(session.status)}：${session.target_position}，${difficultyLabel(session.difficulty)}难度，第 ${session.current_question_index} 轮`" @click="router.push(`/interviews/${session.id}`)">
              <span class="session-icon">{{ session.target_position.slice(0, 1) }}</span>
              <span class="session-main">
                <strong>{{ session.target_position }}</strong>
                <small>{{ difficultyLabel(session.difficulty) }}难度 · 第 {{ session.current_question_index }} 轮</small>
              </span>
              <el-tag :type="statusType(session.status)" effect="light" size="small">{{ statusLabel(session.status) }}</el-tag>
            </button>
          </div>
        </aside>
      </div>
    </section>
  </main>
</template>
