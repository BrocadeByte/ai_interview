<script setup lang="ts">
import { ElEmpty, ElMessage, ElTag, vLoading } from 'element-plus'
import 'element-plus/theme-chalk/el-empty.css'
import 'element-plus/theme-chalk/el-loading.css'
import 'element-plus/theme-chalk/el-message.css'
import 'element-plus/theme-chalk/el-tag.css'
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'

import { getApiErrorMessage } from '../api/client'
import { fetchInterviewReport, fetchReport, type InterviewReport } from '../api/report'

const route = useRoute()
const report = ref<InterviewReport | null>(null)
const loading = ref(false)

const reportId = computed(() => Number(route.params.id))
const sessionId = computed(() => Number(route.params.id))
const isSessionReport = computed(() => route.name === 'interview-report')

function formatScore(value: number) {
  return Number.isFinite(value) ? value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '') : '-'
}

function purposeLabel(purpose: string | null) {
  const labels: Record<string, string> = {
    answer: '回答评分',
    scoring: '评分',
    report: '报告',
  }
  return labels[purpose || ''] || purpose || '知识检索'
}

async function load() {
  loading.value = true
  try {
    const { data } = isSessionReport.value
      ? await fetchInterviewReport(sessionId.value)
      : await fetchReport(reportId.value)
    report.value = data
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error, '报告加载失败'))
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <main class="app-page">
    <section class="content report-detail" v-loading="loading">
      <el-empty v-if="!loading && !report" description="报告不存在" />

      <template v-if="report">
        <header class="report-hero">
          <div>
            <span class="eyebrow">TECH INTERVIEW REVIEW</span>
            <h1>面试复盘报告</h1>
            <p>训练编号 #{{ report.session_id }} · 基于完整问答记录生成</p>
          </div>
          <div class="total-score">
            <span>{{ report.total_score }}</span>
            <small>综合得分</small>
          </div>
        </header>

        <section class="report-section">
          <div class="report-section-title"><span>01</span><h2>总体评价</h2></div>
          <p class="report-summary">{{ report.summary || '暂无总体评价' }}</p>
        </section>

        <section class="report-section">
          <div class="report-section-title"><span>02</span><h2>维度评分</h2></div>
          <el-empty v-if="report.dimension_scores.length === 0" description="暂无维度评分" />
          <div v-else class="dimension-score-list">
            <div v-for="item in report.dimension_scores" :key="item.dimension" class="dimension-score-item">
              <div class="dimension-score-head">
                <div>
                  <strong>{{ item.dimension }}</strong>
                  <small>题目 {{ item.question_indexes.join('、') }}</small>
                </div>
                <span>{{ item.score }}</span>
              </div>
              <div class="score-bar" :aria-label="`${item.dimension} ${item.score} 分`"><div :style="{ width: `${item.score}%` }"></div></div>
              <p>{{ item.focus }}</p>
            </div>
          </div>
        </section>

        <section class="report-grid">
          <div class="report-section">
            <div class="report-section-title success"><span>03</span><h2>核心优势</h2></div>
            <el-empty v-if="report.strengths.length === 0" description="暂无内容" />
            <ul v-else class="report-list-text"><li v-for="item in report.strengths" :key="item">{{ item }}</li></ul>
          </div>
          <div class="report-section">
            <div class="report-section-title warning"><span>04</span><h2>待提升项</h2></div>
            <el-empty v-if="report.weaknesses.length === 0" description="暂无内容" />
            <ul v-else class="report-list-text"><li v-for="item in report.weaknesses" :key="item">{{ item }}</li></ul>
          </div>
        </section>

        <section class="report-grid">
          <div class="report-section">
            <div class="report-section-title"><span>05</span><h2>优化建议</h2></div>
            <el-empty v-if="report.suggestions.length === 0" description="暂无内容" />
            <ul v-else class="report-list-text"><li v-for="item in report.suggestions" :key="item">{{ item }}</li></ul>
          </div>
          <div class="report-section">
            <div class="report-section-title"><span>06</span><h2>学习路线</h2></div>
            <el-empty v-if="report.learning_path.length === 0" description="暂无内容" />
            <ul v-else class="report-list-text"><li v-for="item in report.learning_path" :key="item">{{ item }}</li></ul>
          </div>
        </section>

        <section class="report-section">
          <div class="report-section-title"><span>07</span><h2>示范回答</h2></div>
          <p class="sample-answer">{{ report.sample_answer || '暂无示范回答' }}</p>
        </section>

        <section class="report-section">
          <div class="report-section-title"><span>08</span><h2>知识来源</h2></div>
          <el-empty v-if="report.citations.length === 0" description="该报告生成时未记录知识来源" />
          <div v-else class="citation-list">
            <article
              v-for="citation in report.citations"
              :key="`${citation.purpose}-${citation.question_index}-${citation.document_id}-${citation.index_version}-${citation.chunk_id}-${citation.reference}`"
              class="citation-item"
            >
              <div class="citation-head">
                <div>
                  <strong>{{ citation.title }}</strong>
                  <small>
                    文档 #{{ citation.document_id }} · 版本 {{ citation.index_version }} · chunk {{ citation.chunk_id }}
                    <template v-if="citation.source_page"> · 第 {{ citation.source_page }} 页</template>
                  </small>
                </div>
                <div class="citation-tags">
                  <el-tag size="small" effect="plain">{{ purposeLabel(citation.purpose) }}</el-tag>
                  <el-tag v-if="citation.question_index" size="small" effect="plain">第 {{ citation.question_index }} 题</el-tag>
                  <el-tag v-if="citation.rerank_is_fallback" size="small" type="warning" effect="plain">精排降级</el-tag>
                </div>
              </div>
              <dl class="citation-metrics">
                <div><dt>检索分</dt><dd>{{ formatScore(citation.retrieval_score) }}</dd></div>
                <div><dt>RRF</dt><dd>{{ formatScore(citation.rrf_score) }}</dd></div>
                <div><dt>精排分</dt><dd>{{ formatScore(citation.rerank_score) }}</dd></div>
                <div><dt>召回路线</dt><dd>{{ citation.retrieval_routes.join(' + ') || '-' }}</dd></div>
                <div><dt>上下文 chunks</dt><dd>{{ citation.context_chunks.map(item => item.chunk_id).join(', ') }}</dd></div>
              </dl>
              <p v-if="citation.rerank_fallback_reason" class="citation-warning">{{ citation.rerank_fallback_reason }}</p>
            </article>
          </div>
        </section>
      </template>
    </section>
  </main>
</template>
