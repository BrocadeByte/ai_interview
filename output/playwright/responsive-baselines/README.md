# 响应式截图基线

生成日期：2026-08-18  
浏览器：Playwright Chromium  
视口：桌面 `1440×900`，手机 `390×844`

## 页面矩阵

| 页面 | 路由 | 桌面基线 | 手机基线 |
| --- | --- | --- | --- |
| 登录 | `/login` | `login-desktop-1440x900.png` | `login-mobile-390x844.png` |
| 面试首页 | `/interviews` | `home-desktop-1440x900.png` | `home-mobile-390x844.png` |
| 面试聊天 | `/interviews/42` | `interview-desktop-1440x900.png` | `interview-mobile-390x844.png` |
| 求职画像 | `/profile` | `profile-desktop-1440x900.png` | `profile-mobile-390x844.png` |
| 历史报告 | `/reports` | `reports-desktop-1440x900.png` | `reports-mobile-390x844.png` |
| 报告详情 | `/reports/23` | `report-detail-desktop-1440x900.png` | `report-detail-mobile-390x844.png` |
| 训练前后对比 | `/practice/42/comparison` | `comparison-desktop-1440x900.png` | `comparison-mobile-390x844.png` |
| 知识库 | `/knowledge` | `knowledge-desktop-1440x900.png` | `knowledge-mobile-390x844.png` |

## 基线约定

- 截图使用固定模拟账号与固定业务数据，不依赖真实用户数据、后端数据库或外部模型服务。
- 知识库基线只使用 `is_admin: true` 的管理员身份生成。普通用户不显示知识库导航，直接访问 `/knowledge` 会被重定向到 `/interviews`。
- 基线均为固定视口截图，用于比较首屏布局、固定导航、操作区和响应式换列；长页面的下方内容继续由滚动与溢出检查覆盖。
- 更新基线前需先确认差异来自已审核的界面改动，并同时复核桌面与手机文件。
