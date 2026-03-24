const { createApp } = Vue;

const resolveApiBase = () => {
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:3443`;
};

const FRONTEND_SEEDS = window.PID_FRONTEND_SEEDS || {};

createApp({
  data() {
    return {
      currentPage: 'tuning',
      shellSection: 'tuning-workbench',
      messages: [],
      loading: false,
      loadingMessage: '\u6b63\u5728\u8fde\u63a5...',
      uploadedFile: null,
      dataSource: 'history',
      loopName: 'FIC_101A',
      loopUri: '/pid_zd/5989fb05a2ce4828a7ae36c682906f2b',
      startTime: '1772467200000',
      endTime: '1772553600000',
      historyWindow: 1,
      historyPanelCollapsed: false,
      loopType: 'flow',
      scenario: 'steady_operation',
      plantType: 'distillation_column',
      controlObject: 'reflux_flow',
      experienceLoading: false,
      experienceDetailLoading: false,
      experienceStats: {},
      experienceItems: [],
      selectedExperience: null,
      experienceSearchLoading: false,
      experienceSearchForm: {
        loop_type: 'flow',
        model_type: 'FOPDT',
        K: '',
        T: '',
        L: ''
      },
      experienceSearchResults: null,
      strategyLabForm: {
        candidateId: 'distillation_candidate_v1',
        profileId: 'distillation',
        pluginIds: 'excitation_quality,directional_pair',
        objective: '提升候选窗口排序质量，并保留更丰富的诊断信息。',
        caseId: 'distillation_bidirectional'
      },
      strategyLabCases: [
        {
          id: 'distillation_bidirectional',
          name: '精馏塔塔顶温度双向阶跃案例',
          summary: '包含正负两个方向的阶跃响应，用于检查双向配对和诊断质量。',
          loopType: 'temperature',
          profileId: 'distillation',
          expectedPlugins: ['excitation_quality', 'directional_pair']
        },
        {
          id: 'generic_flow_step',
          name: '通用流量回路单次设定值阶跃案例',
          summary: '适用于默认策略，检查激励质量排序是否合理。',
          loopType: 'flow',
          profileId: 'default',
          expectedPlugins: ['excitation_quality']
        },
        {
          id: 'regulatory_disturbance',
          name: '恒定设定值下的扰动抑制案例',
          summary: '在 SV 基本不变的情况下检查是否可以补充扰动窗口。',
          loopType: 'temperature',
          profileId: 'regulatory',
          expectedPlugins: ['regulatory_window']
        }
      ],
      strategyLabCandidates: [
        {
          id: 'distillation_candidate_demo',
          profileId: 'distillation',
          pluginIds: ['excitation_quality', 'directional_pair'],
          objective: '提升候选窗口排序质量，并保留更丰富的诊断信息。',
          caseId: 'distillation_bidirectional',
          averageScore: 88,
          expectedPluginHit: true,
          bidirectionalPairHit: true,
          releaseGatePassed: true,
          diagnostic: '已命中期望插件，候选窗口的诊断质量较好。',
          benchmarkReport: {
            reportId: 'benchmark-demo-001',
            passedCases: 3,
            totalCases: 3,
            averageScore: 88,
            bestCase: '精馏塔塔顶温度双向阶跃案例',
            mainGap: '抗扰场景下的窗口补充仍可继续增强',
            cases: [
              {
                caseId: 'distillation_bidirectional',
                caseName: '精馏塔塔顶温度双向阶跃案例',
                score: 92,
                passed: true,
                summary: '双向配对命中，诊断信息完整，适合当前精馏场景。'
              },
              {
                caseId: 'generic_flow_step',
                caseName: '通用流量回路单次设定值阶跃案例',
                score: 84,
                passed: true,
                summary: '激励质量排序合理，默认场景下可直接复用。'
              },
              {
                caseId: 'regulatory_disturbance',
                caseName: '恒定设定值下的扰动抑制案例',
                score: 78,
                passed: true,
                summary: '扰动窗口能力基本可用，但还可以继续补强。'
              }
            ]
          },
          codeArtifact: {
            backend: 'heuristic',
            fileName: 'window_plugins/directional_pair.py',
            sourceCode: [
              'def rank_directional_pairs(candidate_windows, diagnostics):',
              '    """',
              '    Prefer paired positive/negative windows for distillation loops.',
              '    Keep explainable diagnostics for the selected pair.',
              '    """',
              '    ranked = []',
              '    for window in candidate_windows:',
              '        score = float(window.get("score", 0.0))',
              '        if window.get("direction") in {"up", "down"}:',
              '            score += 0.15',
              '        ranked.append({**window, "score": score})',
              '    ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)',
              '    diagnostics.append("directional_pair ranking applied")',
              '    return ranked'
            ].join('\n'),
          },
          prompt: [
            '候选策略 ID：精馏塔双向候选策略示例版',
            '策略画像：精馏塔策略',
            '基线案例：精馏塔塔顶温度双向阶跃案例',
            '优化目标：提升候选窗口排序质量，并保留更丰富的诊断信息。',
            '插件列表：激励质量，双向配对',
            '请围绕候选窗口生成、排序与诊断可解释性进行优化。'
          ].join('\n'),
          createdAt: new Date().toLocaleString()
        }
      ],
      selectedStrategyLabCandidateId: 'distillation_candidate_demo',
      strategyLabCompareCandidateId: '',
      experienceDetailDrawerOpen: false,
      strategyLabCandidateDrawerOpen: false,
      strategyLabGenerateModalOpen: false,
      professionalReportDrawerOpen: false,
      systemConfigLoading: false,
      systemConfigSaving: false,
      systemConfigMessage: '',
      systemConfigShowApiKey: false,
      systemConfig: {
        model: {
          name: '',
          api_url: '',
          api_key: ''
        },
        integration: {
          history_data_api_url: '',
          knowledge_graph_api_url: ''
        }
      },
      helpCenterOpen: false,
      helpCenterLoading: false,
      helpCenterHtml: '',
      helpCenterError: '',
      helpCenterToc: [],
      helpCenterActiveSection: '',
      helpCenterExpandedSections: {},
      helpCenterShortcuts: [],
      dataAnalysisExplainOpen: false,
      selectedDataAnalysisExplain: null,
      identificationExplainOpen: false,
      selectedIdentificationExplain: null,
      caseLibraryLoading: false,
      caseLibraryDetailLoading: false,
      caseLibraryDrawerOpen: false,
      caseLibraryStats: {},
      caseLibraryItems: [],
      selectedCase: null,
      caseFilters: {
        provider: '',
        loop_type: '',
        model_type: '',
        track: '',
        failure_mode: '',
        keyword: ''
      },
      experienceFilters: {
        loop_type: '',
        model_type: '',
        passed: '',
        strategy: '',
        keyword: ''
      },
      messageIdCounter: 0,
      compactMode: false,
      executionViewMode: 'all',
      progressSteps: (FRONTEND_SEEDS.progressSteps || [
        { name: '\u6570\u636e\u5206\u6790', icon: '\u6570', active: false, bgActive: 'bg-blue-500 text-white', bgInactive: 'bg-slate-200 text-slate-400' },
        { name: '\u7cfb\u7edf\u8fa8\u8bc6', icon: '\u8fa8', active: false, bgActive: 'bg-green-500 text-white', bgInactive: 'bg-slate-200 text-slate-400' },
        { name: '\u77e5\u8bc6\u589e\u5f3a', icon: '\u77e5', active: false, bgActive: 'bg-cyan-500 text-white', bgInactive: 'bg-slate-200 text-slate-400' },
        { name: 'PID\u6574\u5b9a', icon: '\u6574', active: false, bgActive: 'bg-purple-500 text-white', bgInactive: 'bg-slate-200 text-slate-400' },
        { name: '\u8bc4\u4f30\u53cd\u9988', icon: '\u8bc4', active: false, bgActive: 'bg-orange-500 text-white', bgInactive: 'bg-slate-200 text-slate-400' }
      ]).map(step => ({ ...step }))
    };
  },
      methods: {
        messageStorageKey() {
          return 'pid_tuning_messages_v1';
        },

        strategyLabStorageKey() {
          return 'pid_tuning_strategy_lab_v1';
        },

        saveMessages() {
          localStorage.setItem(this.messageStorageKey(), JSON.stringify({
            messages: this.messages,
            messageIdCounter: this.messageIdCounter,
            compactMode: this.compactMode
          }));
        },

        loadMessages() {
          const raw = localStorage.getItem(this.messageStorageKey());
          if (!raw) return;
          try {
            const parsed = JSON.parse(raw);
            this.messages = Array.isArray(parsed.messages)
              ? parsed.messages.map(msg => ({
                  ...msg,
                  tools: Array.isArray(msg.tools)
                    ? msg.tools.map(tool => ({
                        ...tool,
                        collapsed: Boolean(tool.collapsed)
                      }))
                    : msg.tools
                }))
              : [];
            this.messageIdCounter = Number(parsed.messageIdCounter) || this.messages.reduce((maxId, msg) => Math.max(maxId, msg.id || 0), 0);
            this.compactMode = Boolean(parsed.compactMode);
          } catch (error) {
            console.warn('Failed to restore messages:', error);
          }
        },

        setShellSection(sectionId) {
          if (!sectionId) return;
          this.shellSection = sectionId;
        },

        bindShellSecondaryFallback() {
          const container = document.querySelector('.shell-secondary');
          if (!container || container.dataset.bound === 'true') return;
          container.dataset.bound = 'true';
          container.addEventListener('click', (event) => {
            const button = event.target.closest('[data-section]');
            if (!button) return;
            const sectionId = button.getAttribute('data-section');
            if (sectionId) {
              this.shellSection = sectionId;
            }
          });
        },

        loadStrategyLabState() {
          const raw = localStorage.getItem(this.strategyLabStorageKey());
          if (!raw) return;
          try {
            const parsed = JSON.parse(raw);
            if (parsed.selectedId) {
              this.selectedStrategyLabCandidateId = parsed.selectedId;
            }
            if (parsed.compareId) {
              this.strategyLabCompareCandidateId = parsed.compareId;
            }
          } catch (error) {
            console.warn('Failed to restore strategy lab state:', error);
          }
        },

        saveStrategyLabState() {
          localStorage.setItem(this.strategyLabStorageKey(), JSON.stringify({
            selectedId: this.selectedStrategyLabCandidateId,
            compareId: this.strategyLabCompareCandidateId
          }));
        },

                shellSecondaryItemsFor(page) {
          const map = {
            tuning: [
              { id: 'tuning-workbench', label: '\u5de5\u4f5c\u53f0' },
              { id: 'tuning-history', label: '\u5386\u53f2\u6570\u636e' },
              { id: 'tuning-process', label: '\u6267\u884c\u8fc7\u7a0b' },
              { id: 'tuning-result', label: '\u8c03\u53c2\u7ed3\u679c' },
              { id: 'tuning-explain', label: '\u89e3\u91ca\u8be6\u60c5' }
            ],
            experience: [
              { id: 'experience-overview', label: '\u4e2d\u5fc3\u6982\u89c8' },
              { id: 'experience-search', label: '\u76f8\u4f3c\u68c0\u7d22' },
              { id: 'experience-list', label: '\u7ecf\u9a8c\u5217\u8868' },
              { id: 'experience-detail', label: '\u7ecf\u9a8c\u8be6\u60c5' }
            ],
            'case-library': [
              { id: 'case-overview', label: '\u6848\u4f8b\u6982\u89c8' },
              { id: 'case-list', label: '\u6848\u4f8b\u5217\u8868' },
              { id: 'case-detail', label: '\u6848\u4f8b\u8be6\u60c5' }
            ],
            'strategy-lab': [
              { id: 'strategy-overview', label: '\u5b9e\u9a8c\u6982\u89c8' },
              { id: 'strategy-generate', label: '\u5019\u9009\u751f\u6210' },
              { id: 'strategy-candidates', label: '\u5019\u9009\u5217\u8868' },
              { id: 'strategy-detail', label: '\u5019\u9009\u8be6\u60c5' }
            ],
            'system-config': [
              { id: 'system-model', label: '\u6a21\u578b\u914d\u7f6e' },
              { id: 'system-service', label: '\u670d\u52a1\u63a5\u5165' },
              { id: 'system-summary', label: '\u5f53\u524d\u6458\u8981' }
            ]
          };
          return map[page] || [];
        },

        async loadSystemConfig() {
          this.systemConfigLoading = true;
          this.systemConfigMessage = '';
          try {
            const response = await fetch(`${resolveApiBase()}/api/system-config`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            this.systemConfig = {
              model: {
                name: payload?.model?.name || '',
                api_url: payload?.model?.api_url || '',
                api_key: payload?.model?.api_key || ''
              },
              integration: {
                history_data_api_url: payload?.integration?.history_data_api_url || '',
                knowledge_graph_api_url: payload?.integration?.knowledge_graph_api_url || ''
              }
            };
          } catch (error) {
            console.error('Failed to load system config:', error);
            this.systemConfigMessage = '系统配置加载失败，请检查后端服务。';
          } finally {
            this.systemConfigLoading = false;
          }
        },

        async saveSystemConfig() {
          this.systemConfigSaving = true;
          this.systemConfigMessage = '';
          try {
            const response = await fetch(`${resolveApiBase()}/api/system-config`, {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(this.systemConfig)
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            if (payload?.config) {
              this.systemConfig = payload.config;
            }
            this.systemConfigMessage = '系统配置已保存，新的整定请求会使用最新配置。';
          } catch (error) {
            console.error('Failed to save system config:', error);
            this.systemConfigMessage = '系统配置保存失败，请检查填写内容或后端日志。';
          } finally {
            this.systemConfigSaving = false;
          }
        },

        normalizeStrategyLabCandidate(summary, detail = null) {
          const manifest = detail?.manifest || {};
          const report = detail?.benchmark_report || {};
          const summaryPayload = summary || {};
          const benchmarkSummary = report?.benchmark_summary || manifest?.benchmark_summary || {};
          const releaseGate = report?.release_gate || {};
          const pluginSources = Array.isArray(detail?.plugin_sources) ? detail.plugin_sources : [];
          const primarySource = pluginSources[0] || null;
          const notes = summaryPayload.notes || manifest.notes || '';
          const sourceCandidate = summaryPayload.sourceCandidate
            || summaryPayload.source_candidate
            || manifest.source_candidate
            || detail?.generation_request?.source_candidate
            || (String(notes).startsWith('Cloned from ') ? String(notes).replace('Cloned from ', '').trim() : '');
          return {
            id: summaryPayload.id || manifest.candidate_id,
            profileId: summaryPayload.profile_id || manifest.profile_id || 'default',
            pluginIds: summaryPayload.plugin_ids || manifest.plugin_ids || [],
            status: summaryPayload.status || manifest.status || 'draft',
            objective: summaryPayload.objective || detail?.generation_request?.objective || '',
            caseId: detail?.generation_request?.case_id || 'distillation_bidirectional',
            averageScore: Number(summaryPayload.average_score ?? benchmarkSummary.average_score ?? 0),
            expectedPluginHit: Array.isArray((benchmarkSummary.results || [])[0]?.details?.plugin_hits)
              ? (benchmarkSummary.results[0].details.plugin_hits || []).length > 0
              : Boolean(summaryPayload.release_gate_passed),
            bidirectionalPairHit: Boolean((benchmarkSummary.results || [])[0]?.details?.pair_detected),
            releaseGatePassed: Boolean(summaryPayload.release_gate_passed ?? releaseGate.approved),
            diagnostic: (benchmarkSummary.results || [])[0]?.details?.window_diagnostics?.map(item => this.strategyLabPluginLabel(item.plugin)).join('，') || '暂无诊断摘要',
            prompt: detail?.prompt || '',
            benchmarkReport: detail ? {
              reportId: detail?.manifest?.benchmark_report_path || '',
              passedCases: Number(benchmarkSummary.passed_count || 0),
              totalCases: Number(benchmarkSummary.case_count || 0),
              averageScore: Number(benchmarkSummary.average_score || 0),
              bestCase: this.strategyLabCaseName((benchmarkSummary.results || []).sort((a, b) => Number(b.score || 0) - Number(a.score || 0))[0]?.case_id) || '-',
              mainGap: releaseGate.approved ? '当前候选已通过发布闸门。' : '当前候选尚未通过发布闸门。',
              cases: (benchmarkSummary.results || []).map(item => ({
                caseId: item.case_id,
                caseName: this.strategyLabCaseName(item.case_id),
                score: Number(item.score || 0),
                passed: Boolean(item.passed),
                summary: (item.details?.reason_codes || []).join('；') || item.details?.selection_reason || '已完成候选评测。'
              }))
            } : summaryPayload.benchmarkReport,
            codeArtifact: detail ? {
              backend: detail?.promotion?.generator_backend || 'artifact',
              fileName: primarySource?.file_name || '-',
              sourceCode: primarySource?.source_code || '# 暂无插件源码'
            } : summaryPayload.codeArtifact,
            pluginSources,
            generationRequest: detail?.generation_request || {},
            profile: detail?.profile || {},
            baselineSnapshot: detail?.baseline_snapshot || {},
            promotion: detail?.promotion || {},
            sourceCandidate,
            notes,
            createdAt: summaryPayload.updated_at || ''
          };
        },

        async loadStrategyLabCases() {
          try {
            const response = await fetch(`${resolveApiBase()}/api/strategy-lab/cases`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            this.strategyLabCases = (payload.items || []).map(item => ({
              id: item.id,
              name: item.name,
              profileId: item.profile_id,
              loopType: item.loop_type,
              summary: item.summary,
              expectedPlugins: item.expected_plugins || []
            }));
          } catch (error) {
            console.error('Failed to load strategy lab cases:', error);
          }
        },

        async loadStrategyLabCandidates() {
          try {
            const response = await fetch(`${resolveApiBase()}/api/strategy-lab/candidates`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            this.strategyLabCandidates = (payload.items || []).map(item => this.normalizeStrategyLabCandidate(item));
            if (
              this.strategyLabCandidates.length &&
              (!this.selectedStrategyLabCandidateId || !this.strategyLabCandidates.find(item => item.id === this.selectedStrategyLabCandidateId))
            ) {
              this.selectedStrategyLabCandidateId = this.strategyLabCandidates[0].id;
            }
            if (this.selectedStrategyLabCandidateId) {
              await this.selectStrategyLabCandidate(this.selectedStrategyLabCandidateId);
            }
            if (this.strategyLabCompareCandidateId && !this.strategyLabCandidates.find(item => item.id === this.strategyLabCompareCandidateId)) {
              this.strategyLabCompareCandidateId = '';
            }
            this.saveStrategyLabState();
          } catch (error) {
            console.error('Failed to load strategy lab candidates:', error);
          }
        },

        async switchPage(page) {
          this.currentPage = page;
          const firstSection = this.shellSecondaryItemsFor(page)[0];
          this.shellSection = firstSection ? firstSection.id : '';
          if (page === 'experience') {
            await this.loadExperienceCenter();
            this.shellSection = 'experience-list';
          }
          if (page === 'case-library') {
            await this.loadCaseLibraryCenter();
          }
          if (page === 'strategy-lab') {
            await this.loadStrategyLabCases();
            await this.loadStrategyLabCandidates();
          }
          if (page === 'system-config') {
            await this.loadSystemConfig();
          }
        },

        resetStrategyLabForm() {
          this.strategyLabForm = {
            candidateId: `candidate_${Date.now()}`,
            profileId: 'distillation',
            pluginIds: 'excitation_quality,directional_pair',
            objective: '提升候选窗口排序质量，给出可解释诊断信息。',
            caseId: 'distillation_bidirectional'
          };
        },

        openStrategyLabGenerateModal() {
          this.strategyLabGenerateModalOpen = true;
        },

        closeStrategyLabGenerateModal() {
          this.strategyLabGenerateModalOpen = false;
        },

        async openStrategyLabCandidateDrawer(candidateId = '') {
          const targetId = candidateId || this.selectedStrategyLabCandidateId;
          if (targetId) {
            await this.selectStrategyLabCandidate(targetId);
          }
          this.strategyLabCandidateDrawerOpen = true;
        },

        closeStrategyLabCandidateDrawer() {
          this.strategyLabCandidateDrawerOpen = false;
        },

        async openExperienceDrawer(experienceId = '') {
          if (experienceId) {
            await this.openExperienceDetail(experienceId, false);
          }
          this.experienceDetailDrawerOpen = true;
        },

        closeExperienceDrawer() {
          this.experienceDetailDrawerOpen = false;
        },

        async openCaseLibraryDrawer(caseId = '') {
          if (caseId) {
            await this.openCaseLibraryDetail(caseId, false);
          }
          this.caseLibraryDrawerOpen = true;
        },

        closeCaseLibraryDrawer() {
          this.caseLibraryDrawerOpen = false;
        },

        openProfessionalReportDrawer() {
          this.professionalReportDrawerOpen = true;
        },

        closeProfessionalReportDrawer() {
          this.professionalReportDrawerOpen = false;
        },

        strategyLabCaseName(caseId) {
          return this.strategyLabCases.find(item => item.id === caseId)?.name || caseId || '-';
        },

        strategyLabLoopTypeLabel(loopType) {
          const labels = {
            temperature: '温度',
            flow: '流量',
            pressure: '压力',
            level: '液位'
          };
          return labels[String(loopType || '').trim()] || loopType || '-';
        },

        strategyLabCandidateLabel(candidateId) {
          const labels = {
            example: '默认示例候选策略',
            agent_example: '激励质量候选策略',
            agent_example_v2: '激励质量候选策略 V2',
            distillation_candidate_demo: '精馏塔双向候选策略示例版',
            distillation_candidate_v1: '精馏塔双向候选策略 V1',
            distillation_candidate_v2: '精馏塔双向候选策略 V2',
            distillation_candidate_v3: '精馏塔双向候选策略 V3',
            distillation_candidate_v3_v2: '精馏塔双向候选策略 V3 派生版'
          };
          const raw = String(candidateId || '').trim();
          if (!raw) return '-';
          if (labels[raw]) return labels[raw];
          return raw
            .replace(/^distillation_candidate/i, '精馏塔候选策略')
            .replace(/^agent_example/i, '实验候选策略')
            .replace(/_v(\d+)/gi, ' V$1')
            .replace(/_/g, ' ');
        },

        strategyLabCandidateIdDisplayValue() {
          return this.strategyLabCandidateLabel(this.strategyLabForm.candidateId || '');
        },

        strategyLabCandidateIdFromDisplay(value) {
          const normalized = String(value || '').trim();
          if (!normalized) return '';
          const direct = this.strategyLabCandidates.find(item => this.strategyLabCandidateLabel(item.id) === normalized)?.id;
          if (direct) return direct;
          const preset = {
            '默认示例候选策略': 'example',
            '激励质量候选策略': 'agent_example',
            '激励质量候选策略 V2': 'agent_example_v2',
            '精馏塔双向候选策略示例版': 'distillation_candidate_demo',
            '精馏塔双向候选策略 V1': 'distillation_candidate_v1',
            '精馏塔双向候选策略 V2': 'distillation_candidate_v2',
            '精馏塔双向候选策略 V3': 'distillation_candidate_v3',
            '精馏塔双向候选策略 V3 派生版': 'distillation_candidate_v3_v2'
          };
          return preset[normalized] || normalized;
        },

        updateStrategyLabCandidateIdDisplay(value) {
          this.strategyLabForm.candidateId = this.strategyLabCandidateIdFromDisplay(value);
        },

        strategyLabPluginIdsDisplayValue() {
          return String(this.strategyLabForm.pluginIds || '')
            .split(',')
            .map(item => this.strategyLabPluginLabel(item.trim()))
            .filter(Boolean)
            .join('，');
        },

        updateStrategyLabPluginIdsDisplay(value) {
          const reverse = {
            '激励质量': 'excitation_quality',
            '双向配对': 'directional_pair',
            '抗扰窗口': 'regulatory_window'
          };
          const normalized = String(value || '')
            .replaceAll('，', ',')
            .split(',')
            .map(item => {
              const token = item.trim();
              return reverse[token] || token;
            })
            .filter(Boolean)
            .join(',');
          this.strategyLabForm.pluginIds = normalized;
        },

        strategyLabPluginLabel(pluginId) {
          const labels = {
            excitation_quality: '激励质量',
            directional_pair: '双向配对',
            regulatory_window: '抗扰窗口'
          };
          return labels[pluginId] || pluginId || '-';
        },

        strategyLabGeneratorLabel(backend) {
          const labels = {
            heuristic: '内置启发式生成',
            artifact: '离线实验产物',
            command: '外部命令生成'
          };
          return labels[backend] || backend || '-';
        },

        strategyLabFileLabel(fileName) {
          const value = String(fileName || '').trim();
          if (!value) return 'window_plugin.py';
          if (value.includes('/')) {
            const [folder, name] = value.split(/\/(.+)/, 2);
            if (folder && name) {
              return `${folder} 目录 / ${name}`;
            }
          }
          return value;
        },

        strategyLabPrimaryPlugin(candidate) {
          const pluginId = candidate?.pluginIds?.[0];
          return this.strategyLabPluginLabel(pluginId);
        },

        strategyLabStatusLabel(status) {
          const labels = {
            draft: '草稿',
            validated: '已验证',
            approved: '已批准',
            released: '已发布'
          };
          return labels[String(status || '').trim()] || (status || '未标注');
        },

        strategyLabCodePurpose(candidate) {
          const pluginIds = candidate?.pluginIds || [];
          if (pluginIds.includes('directional_pair')) {
            return '用于双向窗口配对与优先级重排';
          }
          if (pluginIds.includes('regulatory_window')) {
            return '用于补充抗扰场景窗口';
          }
          if (pluginIds.includes('excitation_quality')) {
            return '用于评估激励质量并优化窗口排序';
          }
          return '用于生成候选窗口处理逻辑';
        },

        strategyLabCodeSource(candidate) {
          const backend = candidate?.codeArtifact?.backend;
          if (backend === 'artifact') {
            return '来自离线实验产物';
          }
          if (backend === 'command') {
            return '来自外部命令生成';
          }
          if (backend === 'heuristic') {
            return '来自内置启发式生成';
          }
          return '来源未标注';
        },

        strategyLabProfileLabel(profileId) {
          const labels = {
            distillation: '精馏塔策略',
            default: '默认策略',
            regulatory: '抗扰策略'
          };
          return labels[String(profileId || '').trim()] || (profileId || '-');
        },

        strategyLabCompareCandidates(currentId) {
          return this.strategyLabCandidates.filter(item => item.id !== currentId);
        },

        strategyLabScoreDelta(base, target) {
          const baseScore = Number(base?.averageScore || 0);
          const targetScore = Number(target?.averageScore || 0);
          return targetScore - baseScore;
        },

        strategyLabPluginDiff(base, target) {
          const basePlugins = new Set(base?.pluginIds || []);
          const targetPlugins = new Set(target?.pluginIds || []);
          const added = [...targetPlugins].filter(item => !basePlugins.has(item)).map(item => this.strategyLabPluginLabel(item));
          const removed = [...basePlugins].filter(item => !targetPlugins.has(item)).map(item => this.strategyLabPluginLabel(item));
          if (!added.length && !removed.length) {
            return '插件集合一致';
          }
          const parts = [];
          if (added.length) parts.push(`新增 ${added.join('、')}`);
          if (removed.length) parts.push(`移除 ${removed.join('、')}`);
          return parts.join('；');
        },

        strategyLabCompareHeadline(base, target) {
          if (!base || !target) return '请选择一个候选策略进行对比。';
          const delta = this.strategyLabScoreDelta(base, target);
          const targetLabel = this.strategyLabCandidateLabel(target.id);
          if (delta > 0.01) {
            return `${targetLabel} 相比当前候选平均分提升 ${this.formatNumber(delta, 2)} 分。`;
          }
          if (delta < -0.01) {
            return `${targetLabel} 相比当前候选平均分低 ${this.formatNumber(Math.abs(delta), 2)} 分。`;
          }
          return '两版候选的平均分接近，可继续对比插件和逐案例差异。';
        },

        strategyLabCompareCaseRows(base, target) {
          const baseCases = base?.benchmarkReport?.cases || [];
          const targetCases = target?.benchmarkReport?.cases || [];
          const ids = [...new Set([...baseCases.map(item => item.caseId), ...targetCases.map(item => item.caseId)])];
          return ids.map((caseId) => {
            const current = baseCases.find(item => item.caseId === caseId) || null;
            const compared = targetCases.find(item => item.caseId === caseId) || null;
            const delta = Number((compared?.score || 0) - (current?.score || 0));
            return {
              caseId,
              caseName: this.strategyLabCaseName(caseId),
              current,
              compared,
              delta,
              headline: delta > 0.01
                ? '对比候选在该案例上更优'
                : delta < -0.01
                  ? '当前候选在该案例上更优'
                  : '两版候选在该案例上表现接近'
            };
          });
        },

        strategyLabEvolutionChain(candidate) {
          if (!candidate) return [];
          const byId = new Map(this.strategyLabCandidates.map(item => [item.id, item]));
          const chain = [];
          const seen = new Set();
          let cursor = candidate;
          while (cursor && !seen.has(cursor.id)) {
            chain.unshift(cursor);
            seen.add(cursor.id);
            cursor = cursor.sourceCandidate ? byId.get(cursor.sourceCandidate) || null : null;
          }
          return chain.map((item, index) => ({
            ...item,
            stepLabel: `第 ${index + 1} 版`,
            relationLabel: index === 0 ? '初始候选' : `派生自 ${this.strategyLabCandidateLabel(chain[index - 1].id)}`
          }));
        },

        formatStrategyLabPrompt(prompt) {
          const text = String(prompt || '');
          if (!text) return '';
          const valueMap = {
            distillation_column: '精馏塔',
            reflux_ratio_change: '回流比变化',
            distillation_bidirectional: '精馏塔塔顶温度双向阶跃案例',
            generic_flow_step: '通用流量回路单次设定值阶跃案例',
            regulatory_disturbance: '恒定设定值下的扰动抑制案例',
            tower_top_temperature: '塔顶温度',
            distillation: '精馏塔策略',
            default: '默认策略',
            regulatory: '抗扰策略',
            temperature: '温度',
            flow: '流量',
            pressure: '压力',
            level: '液位',
            excitation_quality: '激励质量',
            directional_pair: '双向配对',
            regulatory_window: '抗扰窗口',
            distillation_candidate: '精馏塔候选策略',
            distillation_candidate_demo: '精馏塔双向候选策略示例版',
            distillation_candidate_v1: '精馏塔双向候选策略 V1',
            distillation_candidate_v2: '精馏塔双向候选策略 V2',
            distillation_candidate_v3: '精馏塔双向候选策略 V3',
            distillation_candidate_v3_v2: '精馏塔双向候选策略 V3 派生版',
            agent_example: '激励质量候选策略',
            agent_example_v2: '激励质量候选策略 V2'
          };
          const linePrefixMap = {
            'Candidate ID': '候选策略 ID',
            'Profile ID': '策略画像',
            'Target Plugins': '目标插件',
            'Objective': '优化目标',
            'Target Equipment': '目标设备/装置',
            'Target Loop Type': '目标回路类型',
            'Target Scenario': '目标工况',
            'Case': '基线案例',
            'Baseline summary': '基线摘要',
            'Baseline profile': '基线策略画像',
            'Case count': '案例数量',
            'Passed count': '通过案例数',
            'Average score': '平均分',
            'Design constraints': '设计约束',
            'Benchmark cases to optimize against': '待优化基准评测案例',
            'Output requirements': '输出要求'
          };
          const localizePromptFragment = (value) => {
            let localized = String(value || '');
            Object.entries(valueMap).forEach(([source, target]) => {
              localized = localized.replaceAll(source, target);
            });
            return localized
              .replaceAll('Baseline profile', '基线策略画像')
              .replaceAll('Case count', '案例数量')
              .replaceAll('Passed count', '通过案例数')
              .replaceAll('Average score', '平均分')
              .replaceAll('selected_event_type', '选中事件类型')
              .replaceAll('diagnostic_plugins', '诊断插件')
              .replaceAll('score=', '得分=')
              .replaceAll('passed=True', '通过=是')
              .replaceAll('passed=False', '通过=否')
              .replaceAll('True', '是')
              .replaceAll('False', '否')
              .replaceAll('no explicit gaps captured', '未记录明显缺口')
              .replaceAll('step_down', '下降阶跃')
              .replaceAll('step_up', '上升阶跃')
              .replaceAll('expect_bidirectional_pair', '期望双向配对')
              .replaceAll('expect_bidirectional_pair', '期望双向配对')
              .replaceAll('expect_bi双向配对', '期望双向配对')
              .replaceAll('min_candidate_windows', '最少候选窗口数')
              .replaceAll('gaps', '缺口')
              .replaceAll('context', '上下文')
              .replaceAll('expectations', '期望')
              .replaceAll('profile_id', '策略画像 ID')
              .replaceAll('expected_plugins', '期望插件')
              .replaceAll('loop_type', '回路类型')
              .replaceAll('loop_name', '回路名称')
              .replaceAll('plant_type', '装置类型')
              .replaceAll('scenario', '工况')
              .replaceAll('control_object', '控制对象')
              .replaceAll('tower_section', '塔段')
              .replaceAll('candidate_windows', '候选窗口')
              .replaceAll('selected_event', '选中事件')
              .replaceAll('diagnostics', '诊断信息')
              .replaceAll('plugins/*.py', 'plugins 目录下的插件文件')
              .replaceAll('profile.json', '策略画像文件 profile.json')
              .replaceAll('generation_request.json', '生成请求文件 generation_request.json')
              .replaceAll('prompt.md', '提示词文件 prompt.md')
              .replaceAll('WindowPluginResult', '窗口插件结果对象 WindowPluginResult')
              .replaceAll('Case ', '案例 ')
              .replaceAll('If baseline gaps mention missing pair detection or fallback behavior, address those gaps explicitly in diagnostics.', '如果基线缺口提到缺少双向配对检测或回退行为，请在诊断中明确回应这些问题。');
          };
          return text
            .split('\n')
            .map((line) => {
              const trimmed = line.trim();
              if (!trimmed) return line;
              const entry = Object.entries(linePrefixMap).find(([key]) => trimmed.startsWith(`${key}:`) || trimmed === key);
              if (entry) {
                const [source, target] = entry;
                if (trimmed === source) return target;
                const value = trimmed.slice(source.length + 1).trim();
                const mappedValue = value
                  .split(',')
                  .map((part) => {
                    const token = part.trim();
                    return localizePromptFragment(
                      valueMap[token] || this.strategyLabCaseName(token) || this.strategyLabCandidateLabel(token) || token
                    );
                  })
                  .join('，');
                return `${target}：${mappedValue}`;
              }
              return localizePromptFragment(line)
                .replace('Keep the plugin side-effect free and deterministic.', '保持插件无副作用且结果稳定可复现。')
                .replace('Return WindowPluginResult with updated candidate_windows, selected_event, and diagnostics.', '返回包含候选窗口、选中事件和诊断信息的窗口插件结果对象。')
                .replace('Do not depend on online services.', '不要依赖在线服务。')
                .replace('Prefer ranking or enriching windows over destructive filtering unless the signal is clearly invalid.', '除非信号明确无效，否则优先做窗口排序和补充诊断，不要直接进行破坏性过滤。')
                .replace('Only modify candidate workspace files: profile.json, generation_request.json, prompt.md, and plugins/*.py.', '只允许修改候选工作区文件：策略画像文件、生成请求文件、提示词文件以及 plugins 目录下的插件文件。')
                .replace('Update only the candidate workspace files.', '只更新候选工作区文件。')
                .replace('Implement or override the requested plugin ids in plugins/*.py.', '在 plugins 目录下的插件文件中实现或覆盖所请求的插件。')
                .replace('Add useful diagnostics that explain why windows were re-ranked or preserved.', '补充有解释性的诊断信息，说明为什么候选窗口被重排或保留。')
                .replace('If baseline gaps mention missing pair detection or fallback behavior, address those gaps explicitly in diagnostics.', '如果基线缺口提到缺少双向配对检测或回退行为，请在诊断中明确回应这些问题。');
            })
            .join('\n')
            .replaceAll('expect_bi双向配对', '期望双向配对')
            .replaceAll('If baseline gaps mention missing pair detection or fallback behavior, address those gaps explicitly in diagnostics.', '如果基线缺口提到缺少双向配对检测或回退行为，请在诊断中明确回应这些问题。')
            .replaceAll('If baseline 缺口 mention missing pair detection or fallback behavior, address those 缺口 explicitly in diagnostics.', '如果基线缺口提到缺少双向配对检测或回退行为，请在诊断中明确回应这些问题。');
        },
        async selectStrategyLabCandidate(candidateId) {
          this.selectedStrategyLabCandidateId = candidateId;
          if (this.strategyLabCompareCandidateId === candidateId) {
            this.strategyLabCompareCandidateId = '';
          }
          try {
            const response = await fetch(`${resolveApiBase()}/api/strategy-lab/candidates/${encodeURIComponent(candidateId)}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            const normalized = this.normalizeStrategyLabCandidate(
              this.strategyLabCandidates.find(item => item.id === candidateId) || { id: candidateId },
              payload
            );
            this.strategyLabCandidates = this.strategyLabCandidates.map(item => item.id === candidateId ? normalized : item);
          } catch (error) {
            console.error('Failed to load strategy lab candidate detail:', error);
          }
          if (this.currentPage === 'strategy-lab') {
            this.shellSection = 'strategy-detail';
          }
          this.strategyLabCandidateDrawerOpen = true;
          this.saveStrategyLabState();
        },

        async generateStrategyLabCandidate() {
          const pluginIds = String(this.strategyLabForm.pluginIds || '')
            .split(',')
            .map(item => item.trim())
            .filter(Boolean);
          try {
            const response = await fetch(`${resolveApiBase()}/api/strategy-lab/candidates/generate`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                candidate_id: this.strategyLabForm.candidateId || `candidate_${Date.now()}`,
                profile_id: this.strategyLabForm.profileId,
                plugin_ids: pluginIds,
                objective: this.strategyLabForm.objective,
                case_id: this.strategyLabForm.caseId
              })
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            await this.loadStrategyLabCandidates();
            if (payload.item?.id) {
              this.selectedStrategyLabCandidateId = payload.item.id;
              await this.evaluateStrategyLabCandidate(payload.item.id);
              if (this.currentPage === 'strategy-lab') {
                this.shellSection = 'strategy-detail';
              }
              this.strategyLabGenerateModalOpen = false;
              this.strategyLabCandidateDrawerOpen = true;
            }
          } catch (error) {
            console.error('Failed to generate strategy lab candidate:', error);
          }
        },

        async evaluateStrategyLabCandidate(candidateId) {
          try {
            const response = await fetch(`${resolveApiBase()}/api/strategy-lab/candidates/${encodeURIComponent(candidateId)}/evaluate`, {
              method: 'POST'
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            await this.loadStrategyLabCandidates();
            await this.selectStrategyLabCandidate(candidateId);
          } catch (error) {
            console.error('Failed to evaluate strategy lab candidate:', error);
          }
        },

        async cloneStrategyLabCandidate(candidateId) {
          try {
            const response = await fetch(`${resolveApiBase()}/api/strategy-lab/candidates/${encodeURIComponent(candidateId)}/clone`, {
              method: 'POST'
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            await this.loadStrategyLabCandidates();
            if (payload.item?.id) {
              this.selectedStrategyLabCandidateId = payload.item.id;
              this.strategyLabCompareCandidateId = candidateId;
              await this.selectStrategyLabCandidate(payload.item.id);
              if (this.currentPage === 'strategy-lab') {
                this.shellSection = 'strategy-detail';
              }
              this.strategyLabCandidateDrawerOpen = true;
            }
          } catch (error) {
            console.error('Failed to clone strategy lab candidate:', error);
          }
        },

        getToolEntry(msg, toolName) {
          if (!msg || !Array.isArray(msg.tools)) return null;
          return msg.tools.find(tool => tool && tool.tool_name === toolName) || null;
        },

        getToolResult(msg, toolName) {
          return this.getToolEntry(msg, toolName)?.result || null;
        },

        hasDataAnalysisExplain(msg) {
          if (!msg || msg.agent !== '数据分析智能体') return false;
          const result = this.getToolResult(msg, 'tool_load_data');
          const selectedWindow = result?.selected_window || {};
          const stepEvents = Array.isArray(result?.step_events) ? result.step_events : [];
          return Boolean(stepEvents.length || selectedWindow.rows || selectedWindow.start_index !== undefined);
        },

        describeStepEventType(type) {
          const mapping = {
            step_up: '设定值上升阶跃',
            step_down: '设定值下降阶跃',
            mv_change: '操纵量变化事件',
            full_range: '全量数据窗口'
          };
          return mapping[String(type || '')] || String(type || '候选事件');
        },

        estimateEventTimeRange(result, event) {
          const startTime = result?.window_overview?.start_time;
          const dt = Number(result?.sampling_time);
          const startIdx = Number(event?.start_idx);
          const endIdx = Number(event?.end_idx);
          if (!startTime || !Number.isFinite(dt) || dt <= 0 || !Number.isFinite(startIdx) || !Number.isFinite(endIdx)) {
            return null;
          }
          const base = new Date(startTime.replace(' ', 'T'));
          if (Number.isNaN(base.getTime())) {
            return null;
          }
          const start = new Date(base.getTime() + startIdx * dt * 1000);
          const end = new Date(base.getTime() + endIdx * dt * 1000);
          const pad = (num) => String(num).padStart(2, '0');
          const format = (value) => `${value.getFullYear()}-${pad(value.getMonth() + 1)}-${pad(value.getDate())} ${pad(value.getHours())}:${pad(value.getMinutes())}:${pad(value.getSeconds())}`;
          return {
            singleLine: `${format(start)} -> ${format(end)}`,
            multiLine: `${format(start)}\n-> ${format(end)}`
          };
        },

        eventTimeRange(result, event) {
          if (event?.start_time && event?.end_time) {
            return `${event.start_time} -> ${event.end_time}`;
          }
          const estimated = this.estimateEventTimeRange(result, event);
          if (estimated) {
            return estimated.singleLine;
          }
          if (result?.window_overview?.window_start_time && result?.window_overview?.window_end_time) {
            return `窗口内候选事件，索引 ${event?.start_idx ?? 0} - ${event?.end_idx ?? 0}`;
          }
          return `索引 ${event?.start_idx ?? 0} - ${event?.end_idx ?? 0}`;
        },

        eventTimeRangeDetail(result, event) {
          if (event?.start_time && event?.end_time) {
            return `${event.start_time}\n-> ${event.end_time}`;
          }
          const estimated = this.estimateEventTimeRange(result, event);
          if (estimated) {
            return estimated.multiLine;
          }
          return this.eventTimeRange(result, event);
        },

        buildDataAnalysisExplainPayload(msg) {
          const result = this.getToolResult(msg, 'tool_load_data');
          if (!result) return null;

          const selectedWindow = result.selected_window || {};
          const windowOverview = result.window_overview || {};
          const stepEvents = Array.isArray(result.step_events) ? result.step_events : [];
          const selectedStart = Number(selectedWindow.event_start_index);
          const selectedEnd = Number(selectedWindow.event_end_index);
          const selectedOrder = stepEvents.findIndex(
            event => Number(event?.start_idx) === selectedStart && Number(event?.end_idx) === selectedEnd
          );
          const selectionReason =
            selectedOrder >= 0
              ? `当前窗口由候选事件 #${selectedOrder + 1} 触发。系统会围绕所有候选阶跃构造窗口，并默认选择阶跃幅值最大的候选窗口进入辨识，因此这条事件被作为本次辨识窗口的来源。`
              : '系统会围绕检测到的候选阶跃构造多个窗口，并默认选择最有代表性的候选窗口进入后续辨识。';

          return {
            windowRows: selectedWindow.rows ?? '-',
            windowTimeRange: this.rangeLabel(
              windowOverview.window_start_time,
              windowOverview.window_end_time,
              selectedWindow.start_index,
              selectedWindow.end_index
            ),
            windowIndexRange: `${selectedWindow.start_index ?? 0} -> ${selectedWindow.end_index ?? 0}`,
            triggerEventLabel: this.describeStepEventType(selectedWindow.event_type),
            selectionReason,
            stepEvents: stepEvents.map((event, idx) => ({
              order: idx + 1,
              isSelected:
                Number(event?.start_idx) === selectedStart &&
                Number(event?.end_idx) === selectedEnd,
              typeLabel: this.describeStepEventType(event?.type),
              indexRange: `${event?.start_idx ?? 0} -> ${event?.end_idx ?? 0}`,
              timeRange: this.eventTimeRange(result, event),
              timeRangeDetail: this.eventTimeRangeDetail(result, event),
              amplitude: this.formatNumber(event?.amplitude, 3),
              svChange:
                Number.isFinite(Number(event?.sv_start)) && Number.isFinite(Number(event?.sv_end))
                  ? `${this.formatNumber(event?.sv_start, 3)} -> ${this.formatNumber(event?.sv_end, 3)}`
                  : '-'
            }))
          };
        },

        openDataAnalysisExplain(msg) {
          const payload = this.buildDataAnalysisExplainPayload(msg);
          if (!payload) return;
          this.selectedDataAnalysisExplain = payload;
          this.dataAnalysisExplainOpen = true;
        },

        closeDataAnalysisExplain() {
          this.dataAnalysisExplainOpen = false;
          this.selectedDataAnalysisExplain = null;
        },

        hasIdentificationExplain(msg) {
          const result = this.getToolResult(msg, 'tool_fit_fopdt');
          return Boolean(result && Array.isArray(result.attempts) && result.attempts.length);
        },

        formatAttemptExplanation(attempt) {
          if (attempt?.success === false) {
            return `该候选拟合失败：${attempt?.error || '模型拟合未成功'}`;
          }
          return `系统在窗口 ${attempt?.window_source || '-'} 上拟合 ${attempt?.model_type || '-'} 模型，得到标准化RMSE=${this.formatNumber(attempt?.normalized_rmse, 3)}、R²=${this.formatNumber(attempt?.r2_score, 3)}、置信度=${this.formatPercent(attempt?.confidence, 1)}，并结合该模型对应的候选整定性能评分进行比较。`;
        },

        buildParameterExplanation(modelType, selectedModelParams) {
          const normalized = String(modelType || '').toUpperCase();
          if (normalized === 'SOPDT') {
            return `当前选择的是 SOPDT 模型。K 表示过程增益，T1 与 T2 表示两个主导惯性时间常数，L 表示死区时间。这组参数来自当前最优候选窗口上的二阶加纯滞后模型拟合结果，系统会把不同模型和窗口的拟合质量与后续整定可用性综合比较后，再确定是否采用这组参数。`;
          }
          if (normalized === 'FOPDT') {
            return `当前选择的是 FOPDT 模型。K 表示过程增益，T 表示主导时间常数，L 表示死区时间。这组参数来自当前最优候选窗口上的一阶加纯滞后模型拟合结果，并经过候选模型比较后确定。`;
          }
          if (normalized === 'FO') {
            return `当前选择的是 FO 模型。K 表示过程增益，T 表示主导时间常数。系统在多个候选窗口和模型中比较后，认为这组一阶模型最能解释当前数据。`;
          }
          if (normalized === 'IPDT') {
            return `当前选择的是 IPDT 模型。K 表示积分增益，L 表示等效死区。系统在候选窗口上比较积分对象与自衡对象的拟合效果后，认为当前对象更符合积分过程特征。`;
          }
          return `这组参数来自当前最优候选窗口上的模型拟合结果，系统会综合比较候选模型的拟合误差、解释度、置信度以及后续整定表现后再最终确定。`;
        },

        buildIdentificationExplainPayload(msg) {
          const result = this.getToolResult(msg, 'tool_fit_fopdt');
          if (!result) return null;

          const modelType = String(result.model_type || '').toUpperCase() || '-';
          const selectedModelParams = result.selected_model_params || {};
          const attempts = Array.isArray(result.attempts) ? result.attempts.slice() : [];
          const sortedAttempts = attempts
            .map((attempt, idx) => ({ ...attempt, _idx: idx }))
            .sort((a, b) => {
              const aSuccess = a.success === false ? 0 : 1;
              const bSuccess = b.success === false ? 0 : 1;
              if (aSuccess !== bSuccess) return bSuccess - aSuccess;
              const perfDiff = Number(b.benchmark_performance_score || 0) - Number(a.benchmark_performance_score || 0);
              if (Math.abs(perfDiff) > 1e-9) return perfDiff;
              return Number(b.confidence || 0) - Number(a.confidence || 0);
            })
            .map((attempt, idx) => ({
              rank: idx + 1,
              isSelected:
                String(attempt.model_type || '').toUpperCase() === modelType &&
                String(attempt.window_source || '') === String(result.selected_window_source || ''),
              modelType: String(attempt.model_type || '').toUpperCase() || '-',
              windowSource: attempt.window_source || '-',
              points: attempt.points ?? '-',
              normalizedRmse: this.formatNumber(attempt.normalized_rmse, 3),
              r2: this.formatNumber(attempt.r2_score, 3),
              confidence: this.formatPercent(attempt.confidence, 1),
              performance: this.formatNumber(attempt.benchmark_performance_score, 2),
              explanation: this.formatAttemptExplanation(attempt)
            }));

          return {
            modelType,
            modelParamsSummary: this.summarizeModelParams({ model_type: modelType, ...selectedModelParams }),
            windowSource: result.selected_window_source || '-',
            confidence: this.formatPercent(result.confidence, 1),
            selectionReason: result.model_selection_reason || '系统会在候选窗口上比较多种模型的拟合质量与后续整定表现，最终选取综合表现最优的模型。',
            parameterExplanation: this.buildParameterExplanation(modelType, selectedModelParams),
            attempts: sortedAttempts
          };
        },

        openIdentificationExplain(msg) {
          const payload = this.buildIdentificationExplainPayload(msg);
          if (!payload) return;
          this.selectedIdentificationExplain = payload;
          this.identificationExplainOpen = true;
        },

        closeIdentificationExplain() {
          this.identificationExplainOpen = false;
          this.selectedIdentificationExplain = null;
        },

        async openHelpCenter() {
          this.helpCenterOpen = true;
          if (this.helpCenterLoading) return;

          this.helpCenterLoading = true;
          this.helpCenterError = '';
          try {
            const response = await fetch(`./help-center.md?v=${Date.now()}`, { cache: 'no-store' });
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            const markdown = await response.text();
            this.renderHelpMarkdown(markdown);
          } catch (error) {
            console.error('Failed to load help center:', error);
            this.helpCenterError = `帮助文档加载失败: ${error.message}`;
          } finally {
            this.helpCenterLoading = false;
          }
        },

        closeHelpCenter() {
          this.helpCenterOpen = false;
        },

        renderHelpMarkdown(markdown) {
          const rawHtml = marked.parse(markdown);
          const wrapper = document.createElement('div');
          wrapper.innerHTML = rawHtml;

          const toc = [];
          let headingIndex = 0;
          wrapper.querySelectorAll('h1, h2, h3').forEach(node => {
            headingIndex += 1;
            const id = `help-section-${headingIndex}`;
            node.id = id;
            toc.push({
              id,
              text: (node.textContent || '').trim(),
              level: Number(node.tagName.substring(1)),
            });
          });

          const expandedSections = {};
          toc.forEach((item, index) => {
            const next = toc[index + 1];
            item.hasChildren = Boolean(next && next.level > item.level);
            if (item.level === 1) {
              expandedSections[item.id] = true;
            }
          });

          const shortcutLabels = {
            '系统定位': '系统定位',
            '多智能体协作流程': '协作流程',
            '当前评分机制': '评分机制',
            '经验机制': '经验机制',
            '当前后端代码结构': '代码结构',
            '主要接口': '接口说明',
          };
          const shortcuts = toc
            .filter(item => item.level <= 2)
            .filter(item => shortcutLabels[item.text])
            .map(item => ({ id: item.id, label: shortcutLabels[item.text] }));

          this.helpCenterHtml = wrapper.innerHTML;
          this.helpCenterToc = toc;
          this.helpCenterExpandedSections = expandedSections;
          this.helpCenterShortcuts = shortcuts;
          this.helpCenterActiveSection = toc[0]?.id || '';
          this.$nextTick(() => this.onHelpScroll());
        },

        scrollHelpTo(sectionId) {
          const container = this.$refs.helpContentBox;
          if (!container) return;
          const target = container.querySelector(`#${sectionId}`);
          if (!target) return;
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          this.helpCenterActiveSection = sectionId;
        },

        onHelpScroll() {
          const container = this.$refs.helpContentBox;
          if (!container || !this.helpCenterToc.length) return;

          const headings = this.helpCenterToc
            .map(item => ({
              id: item.id,
              top: container.querySelector(`#${item.id}`)?.getBoundingClientRect().top ?? Number.POSITIVE_INFINITY,
            }))
            .filter(item => Number.isFinite(item.top));

          const current = headings
            .filter(item => item.top <= 140)
            .sort((a, b) => b.top - a.top)[0] || headings[0];

          if (current?.id) {
            this.helpCenterActiveSection = current.id;
          }
        },

        toggleHelpToc(sectionId) {
          this.helpCenterExpandedSections = {
            ...this.helpCenterExpandedSections,
            [sectionId]: !this.helpCenterExpandedSections[sectionId],
          };
        },

        isHelpTocExpanded(sectionId) {
          return Boolean(this.helpCenterExpandedSections[sectionId]);
        },

        isHelpTocVisible(item) {
          if (item.level === 1) return true;
          const parent = this.findHelpTocParent(item);
          if (!parent) return true;
          return this.isHelpTocExpanded(parent.id);
        },

        findHelpTocParent(targetItem) {
          const idx = this.helpCenterToc.findIndex(item => item.id === targetItem.id);
          if (idx <= 0) return null;
          for (let i = idx - 1; i >= 0; i -= 1) {
            if (this.helpCenterToc[i].level < targetItem.level) {
              return this.helpCenterToc[i];
            }
          }
          return null;
        },

        async openExperienceCenterFromResult(result) {
          const model = result?.model || {};
          this.experienceSearchForm = {
            loop_type: result?.pidParams?.loopType || this.loopType || 'flow',
            model_type: model.modelType || 'FOPDT',
            K: Number.isFinite(model.K) ? String(model.K) : '',
            T: Number.isFinite(model.T) ? String(model.T) : '',
            L: Number.isFinite(model.L) ? String(model.L) : ''
          };
          await this.switchPage('experience');
          this.shellSection = 'experience-search';
          if (this.experienceSearchForm.K && this.experienceSearchForm.T) {
            await this.searchSimilarExperiences();
          }
        },

        async loadExperienceCenter() {
          await Promise.all([
            this.loadExperienceStats(),
            this.loadExperienceItems()
          ]);
        },

        async loadCaseLibraryCenter() {
          await Promise.all([
            this.loadCaseLibraryStats(),
            this.loadCaseLibraryItems()
          ]);
        },

        async loadCaseLibraryStats() {
          try {
            const response = await fetch(`${resolveApiBase()}/api/case-library/stats`);
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.caseLibraryStats = await response.json();
          } catch (error) {
            console.error('Failed to load case library stats:', error);
            this.caseLibraryStats = {};
          }
        },

        async loadCaseLibraryItems() {
          this.caseLibraryLoading = true;
          try {
            const params = new URLSearchParams();
            if (this.caseFilters.provider) params.set('provider', this.caseFilters.provider);
            if (this.caseFilters.loop_type) params.set('loop_type', this.caseFilters.loop_type);
            if (this.caseFilters.model_type) params.set('model_type', this.caseFilters.model_type);
            if (this.caseFilters.track) params.set('track', this.caseFilters.track);
            if (this.caseFilters.failure_mode) params.set('failure_mode', this.caseFilters.failure_mode);
            if (this.caseFilters.keyword) params.set('keyword', this.caseFilters.keyword);
            params.set('limit', '100');

            const response = await fetch(`${resolveApiBase()}/api/case-library?${params.toString()}`);
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            const payload = await response.json();
            this.caseLibraryItems = Array.isArray(payload.items) ? payload.items : [];
            if (this.selectedCase) {
              const stillExists = this.caseLibraryItems.some(item => item.case_id === this.selectedCase.case_id);
              if (!stillExists) {
                this.selectedCase = null;
              }
            }
            if (!this.selectedCase && this.caseLibraryItems.length) {
              await this.openCaseLibraryDetail(this.caseLibraryItems[0].case_id, false);
            }
          } catch (error) {
            console.error('Failed to load case library items:', error);
            this.caseLibraryItems = [];
          } finally {
            this.caseLibraryLoading = false;
          }
        },

        async openCaseLibraryDetail(caseId, switchSection = true) {
          if (!caseId) return;
          this.caseLibraryDetailLoading = true;
          try {
            const response = await fetch(`${resolveApiBase()}/api/case-library/${encodeURIComponent(caseId)}`);
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            const payload = await response.json();
            this.selectedCase = payload.item || null;
            if (switchSection && this.currentPage === 'case-library') {
              this.shellSection = 'case-detail';
              this.caseLibraryDrawerOpen = true;
            }
          } catch (error) {
            console.error('Failed to load case detail:', error);
            this.selectedCase = null;
          } finally {
            this.caseLibraryDetailLoading = false;
          }
        },

        async resetCaseFilters() {
          this.caseFilters = {
            provider: '',
            loop_type: '',
            model_type: '',
            track: '',
            failure_mode: '',
            keyword: ''
          };
          await this.loadCaseLibraryItems();
        },

        async clearExperienceCenter() {
          const confirmed = window.confirm('确认清除全部历史经验吗？此操作便于测试使用，清除后不可恢复。');
          if (!confirmed) return;

          try {
            const response = await fetch(`${resolveApiBase()}/api/experiences/actions/clear`, {
              method: 'POST'
            });
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.selectedExperience = null;
            this.experienceSearchResults = null;
            await this.loadExperienceCenter();
          } catch (error) {
            console.error('Failed to clear experiences:', error);
            alert(`清除历史经验失败: ${error.message}`);
          }
        },

        async loadExperienceStats() {
          try {
            const response = await fetch(`${resolveApiBase()}/api/experiences/stats`);
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            const payload = await response.json();
            this.experienceStats = {
              ...payload,
              top_strategy: payload.top_strategy || payload.top_strategies?.[0]?.strategy || ''
            };
          } catch (error) {
            console.error('Failed to load experience stats:', error);
            this.experienceStats = {};
          }
        },

        async loadExperienceItems() {
          this.experienceLoading = true;
          try {
            const params = new URLSearchParams();
            if (this.experienceFilters.loop_type) params.set('loop_type', this.experienceFilters.loop_type);
            if (this.experienceFilters.model_type) params.set('model_type', this.experienceFilters.model_type);
            if (this.experienceFilters.passed) params.set('passed', this.experienceFilters.passed);
            if (this.experienceFilters.strategy) params.set('strategy', this.experienceFilters.strategy);
            if (this.experienceFilters.keyword) params.set('keyword', this.experienceFilters.keyword);
            params.set('limit', '100');

            const response = await fetch(`${resolveApiBase()}/api/experiences?${params.toString()}`);
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            const payload = await response.json();
            this.experienceItems = Array.isArray(payload.items) ? payload.items.map(item => ({
              ...item,
              final_strategy: item.final_strategy || item.strategy?.final || '',
              final_rating: item.final_rating ?? item.evaluation?.final_rating ?? 0,
              performance_score: item.performance_score ?? item.evaluation?.performance_score ?? 0,
              passed: item.passed ?? item.evaluation?.passed ?? false,
              model_type: item.model_type || item.model?.model_type || 'FOPDT',
              K: item.K ?? item.model?.K ?? 0,
              T: item.T ?? item.model?.T ?? 0,
              L: item.L ?? item.model?.L ?? 0,
            })) : [];
            if (this.selectedExperience) {
              const stillExists = this.experienceItems.some(item => item.experience_id === this.selectedExperience.experience_id);
              if (!stillExists) {
                this.selectedExperience = null;
              }
            }
            if (!this.selectedExperience && this.experienceItems.length) {
              await this.openExperienceDetail(this.experienceItems[0].experience_id, false);
            }
          } catch (error) {
            console.error('Failed to load experiences:', error);
            this.experienceItems = [];
          } finally {
            this.experienceLoading = false;
          }
        },

        async openExperienceDetail(experienceId, switchSection = true) {
          if (!experienceId) return;
          this.experienceDetailLoading = true;
          try {
            const response = await fetch(`${resolveApiBase()}/api/experiences/${experienceId}`);
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            const payload = await response.json();
            this.selectedExperience = payload.item || null;
            if (switchSection && this.currentPage === 'experience') {
              this.shellSection = 'experience-detail';
              this.experienceDetailDrawerOpen = true;
            }
          } catch (error) {
            console.error('Failed to load experience detail:', error);
            this.selectedExperience = null;
          } finally {
            this.experienceDetailLoading = false;
          }
        },

        async resetExperienceFilters() {
          this.experienceFilters = {
            loop_type: '',
            model_type: '',
            passed: '',
            strategy: '',
            keyword: ''
          };
          await this.loadExperienceItems();
        },

        async searchSimilarExperiences() {
          const K = Number(this.experienceSearchForm.K);
          const T = Number(this.experienceSearchForm.T);
          const L = Number(this.experienceSearchForm.L || 0);
          if (!Number.isFinite(K) || !Number.isFinite(T)) {
            this.experienceSearchResults = {
              matches: [],
              guidance: '请输入有效的 K 和 T 后再检索相似经验。'
            };
            return;
          }

          this.experienceSearchLoading = true;
          try {
            const formData = new FormData();
            formData.append('loop_type', this.experienceSearchForm.loop_type || 'flow');
            formData.append('model_type', this.experienceSearchForm.model_type || 'FOPDT');
            formData.append('K', String(K));
            formData.append('T', String(T));
            formData.append('L', String(Number.isFinite(L) ? L : 0));
            formData.append('limit', '3');

            const response = await fetch(`${resolveApiBase()}/api/experiences/search`, {
              method: 'POST',
              body: formData
            });
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.experienceSearchResults = await response.json();
          } catch (error) {
            console.error('Failed to search similar experiences:', error);
            this.experienceSearchResults = {
              matches: [],
              guidance: `相似经验检索失败: ${error.message}`
            };
          } finally {
            this.experienceSearchLoading = false;
          }
        },

        handleFileUpload(event) {
          this.uploadedFile = event.target.files[0];
        },

        clearUploadedFile() {
          this.uploadedFile = null;
          if (this.$refs.fileInput) {
            this.$refs.fileInput.value = '';
          }
        },

        progressStepLabel(step, idx) {
          const fallback = ['\u6570\u636e\u5206\u6790', '\u7cfb\u7edf\u8fa8\u8bc6', '\u77e5\u8bc6\u589e\u5f3a', 'PID\u6574\u5b9a', '\u8bc4\u4f30\u53cd\u9988'];
          const value = (step?.name || '').trim();
          return value && !/^\?+$/.test(value) && !value.includes('PID??') ? value : (fallback[idx] || '\u6b65\u9aa4');
        },

        progressStepIcon(step, idx) {
          const fallback = ['\u6570', '\u8fa8', '\u77e5', '\u6574', '\u8bc4'];
          const value = (step?.icon || '').trim();
          return value && value !== '?' ? value : (fallback[idx] || '\u2026');
        },

        getAgentIcon(agentName) {
          const icons = {
            '\u6570\u636e\u5206\u6790\u667a\u80fd\u4f53': '\u6570',
            '\u7cfb\u7edf\u8fa8\u8bc6\u667a\u80fd\u4f53': '\u8fa8',
            '\u672c\u4f53\u77e5\u8bc6\u667a\u80fd\u4f53': '\u77e5',
            'PID\u4e13\u5bb6\u667a\u80fd\u4f53': '\u6574',
            '\u8bc4\u4f30\u667a\u80fd\u4f53': '\u8bc4'
          };
          return icons[agentName] || '\u667a';
        },

        getAgentBorderClass(agentName) {
          const classes = {
            '\u6570\u636e\u5206\u6790\u667a\u80fd\u4f53': 'agent-border-blue border-blue-300',
            '\u7cfb\u7edf\u8fa8\u8bc6\u667a\u80fd\u4f53': 'agent-border-green border-green-300',
            '\u672c\u4f53\u77e5\u8bc6\u667a\u80fd\u4f53': 'agent-border-cyan border-cyan-300',
            'PID\u4e13\u5bb6\u667a\u80fd\u4f53': 'agent-border-purple border-purple-300',
            '\u8bc4\u4f30\u667a\u80fd\u4f53': 'agent-border-orange border-orange-300'
          };
          return classes[agentName] || 'border-slate-300';
        },

        getAgentBgClass(agentName) {
          const classes = {
            '\u6570\u636e\u5206\u6790\u667a\u80fd\u4f53': 'bg-blue-50',
            '\u7cfb\u7edf\u8fa8\u8bc6\u667a\u80fd\u4f53': 'bg-green-50',
            '\u672c\u4f53\u77e5\u8bc6\u667a\u80fd\u4f53': 'bg-cyan-50',
            'PID\u4e13\u5bb6\u667a\u80fd\u4f53': 'bg-purple-50',
            '\u8bc4\u4f30\u667a\u80fd\u4f53': 'bg-orange-50'
          };
          return classes[agentName] || 'bg-slate-50';
        },

        getAgentTextClass(agentName) {
          const classes = {
            '\u6570\u636e\u5206\u6790\u667a\u80fd\u4f53': 'text-blue-800',
            '\u7cfb\u7edf\u8fa8\u8bc6\u667a\u80fd\u4f53': 'text-green-800',
            '\u672c\u4f53\u77e5\u8bc6\u667a\u80fd\u4f53': 'text-cyan-800',
            'PID\u4e13\u5bb6\u667a\u80fd\u4f53': 'text-purple-800',
            '\u8bc4\u4f30\u667a\u80fd\u4f53': 'text-orange-800'
          };
          return classes[agentName] || 'text-slate-800';
        },

        getAgentColor(agentName) {
          const colors = {
            '\u6570\u636e\u5206\u6790\u667a\u80fd\u4f53': { bg: '#eff6ff', border: '#3b82f6', text: '#1e40af' },
            '\u7cfb\u7edf\u8fa8\u8bc6\u667a\u80fd\u4f53': { bg: '#f0fdf4', border: '#10b981', text: '#065f46' },
            '\u672c\u4f53\u77e5\u8bc6\u667a\u80fd\u4f53': { bg: '#ecfeff', border: '#06b6d4', text: '#155e75' },
            'PID\u4e13\u5bb6\u667a\u80fd\u4f53': { bg: '#faf5ff', border: '#8b5cf6', text: '#6b21a8' },
            '\u8bc4\u4f30\u667a\u80fd\u4f53': { bg: '#fff7ed', border: '#f59e0b', text: '#92400e' }
          };
          return colors[agentName] || { bg: '#f8fafc', border: '#cbd5e1', text: '#475569' };
        },

        updateProgress(agentName) {
          const agentMap = {
            '\u6570\u636e\u5206\u6790\u667a\u80fd\u4f53': 0,
            '\u7cfb\u7edf\u8fa8\u8bc6\u667a\u80fd\u4f53': 1,
            '\u672c\u4f53\u77e5\u8bc6\u667a\u80fd\u4f53': 2,
            'PID\u4e13\u5bb6\u667a\u80fd\u4f53': 3,
            '\u8bc4\u4f30\u667a\u80fd\u4f53': 4
          };
          const index = agentMap[agentName];
          if (index !== undefined) {
            this.progressSteps.forEach((step, i) => {
              step.active = i === index;
            });
          }
        },

        async consumeSSEStream(response) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

            let boundaryIndex = buffer.indexOf('\n\n');
            while (boundaryIndex !== -1) {
              const rawEvent = buffer.slice(0, boundaryIndex);
              buffer = buffer.slice(boundaryIndex + 2);

              const dataLines = rawEvent
                .split('\n')
                .filter(line => line.startsWith('data: '))
                .map(line => line.substring(6));

              if (dataLines.length) {
                const payload = dataLines.join('\n');
                const data = JSON.parse(payload);
                this.handleSSEMessage(data);
              }

              boundaryIndex = buffer.indexOf('\n\n');
            }

            if (done) {
              const trailing = buffer.trim();
              if (trailing) {
                const dataLines = trailing
                  .split('\n')
                  .filter(line => line.startsWith('data: '))
                  .map(line => line.substring(6));
                if (dataLines.length) {
                  const payload = dataLines.join('\n');
                  const data = JSON.parse(payload);
                  this.handleSSEMessage(data);
                }
              }
              break;
            }
          }
        },

        async startTuning() {
          if (this.loading) return;

          const usingUploadedCsv = this.dataSource === 'upload';
          if (usingUploadedCsv && !this.uploadedFile) {
            this.addMessage({
              type: 'assistant',
              content: '请选择CSV文件后再开始整定。'
            });
            return;
          }

          this.loading = true;
          this.loadingMessage = usingUploadedCsv ? '正在上传文件...' : '正在获取历史数据...';

          this.addMessage({
            type: 'user',
            content: usingUploadedCsv
              ? `为控制回路 ${this.loopName} 整定PID参数
数据来源: 上传CSV
回路类型: ${this.loopType}
工况: ${this.scenarioLabel(this.scenario)}
装置类型: ${this.plantTypeLabel(this.plantType)}
控制对象: ${this.controlObjectLabel(this.controlObject)}`
              : `为控制回路 ${this.loopName} 整定PID参数
数据来源: 历史数据
loop_uri: ${this.loopUri || '/pid_zd/5989fb05a2ce4828a7ae36c682906f2b'}
start_time: ${this.startTime || '1772467200000'}
end_time: ${this.endTime || '1772553600000'}
window: ${this.historyWindow || 1}
回路类型: ${this.loopType}
工况: ${this.scenarioLabel(this.scenario)}
装置类型: ${this.plantTypeLabel(this.plantType)}
控制对象: ${this.controlObjectLabel(this.controlObject)}`,
            file_name: usingUploadedCsv ? this.uploadedFile.name : null
          });

          const formData = new FormData();
          if (usingUploadedCsv) {
            formData.append('file', this.uploadedFile);
          }
          formData.append('loop_name', this.loopName);
          formData.append('loop_type', this.loopType);
          formData.append('loop_uri', this.loopUri || '/pid_zd/5989fb05a2ce4828a7ae36c682906f2b');
          formData.append('start_time', this.startTime || '1772467200000');
          formData.append('end_time', this.endTime || '1772553600000');
          formData.append('window', String(this.historyWindow || 1));
          formData.append('scenario', this.scenario || '');
          formData.append('plant_type', this.plantType || 'distillation_column');
          formData.append('control_object', this.controlObject || '');

          try {
            this.loadingMessage = '智能体正在协同处理...';
            const response = await fetch(`${resolveApiBase()}/api/tune_stream`, {
              method: 'POST',
              body: formData
            });
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            await this.consumeSSEStream(response);
          } catch (error) {
            console.error('Tuning failed:', error);
            this.addMessage({
              type: 'assistant',
              content: `请求失败: ${error.message}`
            });
          } finally {
            this.loading = false;
            this.progressSteps.forEach(step => step.active = false);
          }
        },

        handleSSEMessage(data) {
          if (data.type === 'agent_turn') {
            this.updateProgress(data.agent);
            this.addMessage({
              type: 'agent_turn',
              agent: data.agent,
              tools: (data.tools || []).map(tool => ({
                ...tool,
                collapsed: Boolean(tool.collapsed)
              })),
              response: data.response || '',
              collapsed: false
            });
          } else if (data.type === 'thought') {
            this.addMessage({
              type: 'thought',
              agent: data.agent,
              content: data.content,
              collapsed: true
            });
          } else if (data.type === 'tool_call') {
            this.addMessage({
              type: 'tool_call',
              tool_name: data.tool_name,
              args: data.args,
              collapsed: true,
              result: null,
              is_error: false
            });
          } else if (data.type === 'tool_result') {
            const lastToolMsg = [...this.messages].reverse()
              .find(m => m.type === 'tool_call' && !m.result);
            if (lastToolMsg) {
              lastToolMsg.result = data.result;
              lastToolMsg.is_error = data.is_error || false;
              this.saveMessages();
            }
          } else if (data.type === 'assistant') {
            this.addMessage({
              type: 'assistant',
              content: data.content
            });
          } else if (data.type === 'result') {
            this.addMessage({
              type: 'result',
              data: data.data,
              collapsed: false
            });
          } else if (data.type === 'error') {
            this.addMessage({
              type: 'assistant',
              content: `❌ 错误: ${data.message}`
            });
          }

          this.$nextTick(() => {
            const chatBox = this.$refs.chatBox;
            if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
          });
        },

        addMessage(msg) {
          this.messages.push({
            id: ++this.messageIdCounter,
            ...msg
          });
          this.saveMessages();
        },

        toggleCollapse(msgId) {
          const msg = this.messages.find(m => m.id === msgId);
          if (msg) {
            msg.collapsed = !msg.collapsed;
            this.saveMessages();
          }
        },

        toggleToolCollapse(msgId, toolIndex) {
          const msg = this.messages.find(m => m.id === msgId);
          if (!msg || !Array.isArray(msg.tools) || !msg.tools[toolIndex]) {
            return;
          }
          msg.tools[toolIndex].collapsed = !msg.tools[toolIndex].collapsed;
          this.saveMessages();
        },

        setExecutionViewMode(mode) {
          this.executionViewMode = mode || 'all';
        },

        expandAllExecutionCards() {
          this.messages.forEach(msg => {
            if (msg.type === 'agent_turn') {
              msg.collapsed = false;
              if (Array.isArray(msg.tools)) {
                msg.tools.forEach(tool => {
                  tool.collapsed = false;
                });
              }
            }
          });
          this.saveMessages();
        },

        collapseAllExecutionCards() {
          this.messages.forEach(msg => {
            if (msg.type === 'agent_turn') {
              msg.collapsed = false;
              if (Array.isArray(msg.tools)) {
                msg.tools.forEach(tool => {
                  tool.collapsed = true;
                });
              }
            }
          });
          this.saveMessages();
        },

        formatJson(obj) {
          return JSON.stringify(obj, null, 2);
        },

        formatNumber(value, digits = 3) {
          return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-';
        },

        formatPercent(value, digits = 1) {
          return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '-';
        },

        formatScore100(value, digits = 1) {
          return typeof value === 'number' && Number.isFinite(value) ? (value * 10).toFixed(digits) : '-';
        },

        summarizeModelParams(model) {
          if (!model || typeof model !== 'object') {
            return '-';
          }

          const entries = [];
          const modelType = String(model.model_type || model.modelType || '').toUpperCase();
          const orderedKeys =
            modelType === 'SOPDT'
              ? ['K', 'T1', 'T2', 'L']
              : modelType === 'IPDT'
                ? ['K', 'Ki_proc', 'L']
                : ['K', 'T', 'L', 'T1', 'T2', 'Ki_proc'];
          orderedKeys.forEach(key => {
            const value = Number(model[key]);
            if (Number.isFinite(value)) {
              entries.push(`${key}=${value.toFixed(3)}`);
            }
          });

          return entries.length ? entries.join(', ') : '-';
        },

        getModelPrimaryMetrics(model) {
          if (!model || typeof model !== 'object') {
            return [];
          }

          const modelType = String(model.modelType || model.model_type || '').toUpperCase();
          const params = model.selectedModelParams || model.selected_model_params || model;

          if (modelType === 'SOPDT') {
            return [
              { label: '增益 K', value: this.formatNumber(params?.K, 3) },
              { label: '时间常数 T1', value: this.formatNumber(params?.T1, 2), suffix: 's' },
              { label: '时间常数 T2', value: this.formatNumber(params?.T2, 2), suffix: 's' },
              { label: '死区时间 L', value: this.formatNumber(params?.L, 2), suffix: 's' },
            ];
          }

          if (modelType === 'IPDT') {
            return [
              { label: '过程增益 K', value: this.formatNumber(params?.K, 3) },
              { label: '积分增益 Ki_proc', value: this.formatNumber(params?.Ki_proc, 3) },
              { label: '死区时间 L', value: this.formatNumber(params?.L, 2), suffix: 's' },
            ];
          }

          if (modelType === 'FO') {
            return [
              { label: '增益 K', value: this.formatNumber(params?.K, 3) },
              { label: '时间常数 T', value: this.formatNumber(params?.T, 2), suffix: 's' },
            ];
          }

          return [
            { label: '增益 K', value: this.formatNumber(params?.K ?? model.K, 3) },
            { label: '时间常数 T', value: this.formatNumber(params?.T ?? model.T, 2), suffix: 's' },
            { label: '死区时间 L', value: this.formatNumber(params?.L ?? model.L, 2), suffix: 's' },
          ];
        },

        modelParamsDiffer(rawModel, workingModel) {
          const rawSummary = this.summarizeModelParams(rawModel);
          const workingSummary = this.summarizeModelParams(workingModel);
          if (rawSummary === '-' || workingSummary === '-') {
            return false;
          }
          return rawSummary !== workingSummary;
        },

        buildTrendPath(points, key, width = 420, height = 180) {
          if (!Array.isArray(points) || points.length < 2) return '';
          const paddingX = 18;
          const paddingY = 14;
          const usableWidth = width - paddingX * 2;
          const usableHeight = height - paddingY * 2;
          const values = points
            .map(point => Number(point[key]))
            .filter(value => Number.isFinite(value));
          if (!values.length) return '';

          const minValue = Math.min(...values);
          const maxValue = Math.max(...values);
          const valueSpan = Math.max(maxValue - minValue, 1e-6);
          const lastIndex = Math.max(points.length - 1, 1);

          return points.map((point, idx) => {
            const value = Number(point[key]);
            if (!Number.isFinite(value)) {
              return '';
            }
            const x = paddingX + (usableWidth * idx) / lastIndex;
            const y = height - paddingY - ((value - minValue) / valueSpan) * usableHeight;
            return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
          }).filter(Boolean).join(' ');
        },

        windowHighlightRect(overview, width = 420) {
          const paddingX = 18;
          const usableWidth = width - paddingX * 2;
          const totalPoints = Math.max(Number(overview?.total_points) || 0, 1);
          const windowStart = Math.max(Number(overview?.window_start) || 0, 0);
          const windowEnd = Math.max(Number(overview?.window_end) || windowStart, windowStart);
          const startRatio = windowStart / totalPoints;
          const endRatio = windowEnd / totalPoints;
          const x = paddingX + usableWidth * startRatio;
          const rectWidth = Math.max(usableWidth * (endRatio - startRatio), 2);
          return { x, width: rectWidth };
        },

        windowMarkerPositions(overview, width = 420) {
          const paddingX = 18;
          const usableWidth = width - paddingX * 2;
          const totalPoints = Math.max(Number(overview?.total_points) || 1, 1);
          const windowStart = Math.max(Number(overview?.window_start) || 0, 0);
          const windowEnd = Math.max(Number(overview?.window_end) || windowStart, windowStart);
          return {
            start: paddingX + usableWidth * (windowStart / totalPoints),
            end: paddingX + usableWidth * (windowEnd / totalPoints),
          };
        },

        seriesMin(points, key) {
          if (!Array.isArray(points) || !points.length) return null;
          const values = points.map(point => Number(point[key])).filter(value => Number.isFinite(value));
          return values.length ? Math.min(...values) : null;
        },

        seriesMax(points, key) {
          if (!Array.isArray(points) || !points.length) return null;
          const values = points.map(point => Number(point[key])).filter(value => Number.isFinite(value));
          return values.length ? Math.max(...values) : null;
        },

        axisStartLabel(series) {
          if (series?.x_axis === 'timestamp' && series?.start_time) {
            return series.start_time;
          }
          return '起点';
        },

        axisEndLabel(series) {
          if (series?.x_axis === 'timestamp' && series?.end_time) {
            return series.end_time;
          }
          return '终点';
        },

        overallAxisStartLabel(data) {
          const historyStart = data?.dataAnalysis?.historyRange?.startTime;
          if (historyStart) return historyStart;
          return this.axisStartLabel(data?.model?.windowOverview);
        },

        overallAxisEndLabel(data) {
          const historyEnd = data?.dataAnalysis?.historyRange?.endTime;
          if (historyEnd) return historyEnd;
          return this.axisEndLabel(data?.model?.windowOverview);
        },

        windowBoundaryStartLabel(data) {
          const overview = data?.model?.windowOverview;
          return overview?.window_start_time || `索引 ${data?.dataAnalysis?.selectedWindow?.start_index ?? 0}`;
        },

        windowBoundaryEndLabel(data) {
          const overview = data?.model?.windowOverview;
          return overview?.window_end_time || `索引 ${data?.dataAnalysis?.selectedWindow?.end_index ?? 0}`;
        },

        rangeLabel(startTime, endTime, startIndex, endIndex) {
          if (startTime && endTime) {
            return `${startTime} -> ${endTime}`;
          }
          return `${startIndex ?? 0} -> ${endIndex ?? 0}`;
        },

        copyResult(data) {
          const text = `PID参数整定结果\n\n` +
            `系统模型:\n` +
            `  模型类型 = ${data.model?.modelType || 'FOPDT'}\n` +
            `  原始模型参数 = ${this.summarizeModelParams(data.model?.selectedModelParams)}\n` +
            (this.modelParamsDiffer(data.model?.selectedModelParams, data.model?.tuningModel)
              ? `  当前工作模型 = ${this.summarizeModelParams(data.model?.tuningModel)}\n`
              : '') +
            `  选模依据 = ${data.model?.modelSelectionReason || '-'}\n` +
            `  K = ${data.model?.K.toFixed(3)}\n` +
            `  T = ${data.model?.T.toFixed(2)}s\n` +
            `  L = ${data.model?.L.toFixed(2)}s\n` +
            `  置信度 = ${this.formatPercent(data.model?.confidence)}\n` +
            `  标准化RMSE = ${this.formatNumber(data.model?.normalizedRmse, 4)}\n` +
            `  原始RMSE = ${this.formatNumber(data.model?.rawRmse, 4)}\n` +
            `  R² = ${this.formatNumber(data.model?.r2Score, 4)}\n` +
            `  原因 = ${(data.model?.reasonCodes || []).join(', ') || '无'}\n` +
            `  建议动作 = ${(data.model?.nextActions || []).join(', ') || '无'}\n\n` +
            `PID参数:\n` +
            `  Kp = ${data.pidParams?.Kp.toFixed(4)}\n` +
            `  Ki = ${data.pidParams?.Ki.toFixed(4)}\n` +
            `  Kd = ${data.pidParams?.Kd.toFixed(4)}\n` +
            `  策略选择 = ${data.pidParams?.strategyRequested || 'AUTO'}\n` +
            `  实际策略 = ${data.pidParams?.strategyUsed || data.pidParams?.strategy}\n` +
            `  回路类型 = ${data.pidParams?.loopType || '-'}\n` +
            `  历史经验偏好 = ${data.memory?.experienceGuidance?.summary?.preferred_strategy || '-'}\n` +
            `  历史偏好模型 = ${data.memory?.experienceGuidance?.matches?.[0]?.model_type || data.model?.modelType || '-'}\n` +
            `  历史相似案例数 = ${data.memory?.experienceGuidance?.summary?.match_count ?? 0}\n` +
            `  推荐收紧比例 = Kp×${this.formatNumber(data.memory?.experienceGuidance?.summary?.recommended_kp_scale, 2)} / Ki×${this.formatNumber(data.memory?.experienceGuidance?.summary?.recommended_ki_scale, 2)}\n\n` +
            `本体知识参考:\n` +
            `  命中规则数 = ${data.knowledge?.guidance?.matched_count ?? 0}\n` +
            `  推荐策略 = ${data.knowledge?.guidance?.preferred_strategy || '-'}\n` +
            `  命中规则 = ${(data.knowledge?.guidance?.matched_rules || []).map(item => item.title).join('；') || '-'}\n` +
            `  主要约束 = ${(data.knowledge?.guidance?.constraints || []).map(item => item.title).join('；') || '-'}\n` +
            `  专家经验 = ${(data.knowledge?.guidance?.risk_hints || []).join('；') || '-'}\n\n` +
            `${data.evaluation?.initial_assessment?.evaluated_pid && data.evaluation?.auto_refine_result?.applied ? `首次评估PID参数:\n  Kp = ${this.formatNumber(data.evaluation.initial_assessment.evaluated_pid.Kp, 4)}\n  Ki = ${this.formatNumber(data.evaluation.initial_assessment.evaluated_pid.Ki, 4)}\n  Kd = ${this.formatNumber(data.evaluation.initial_assessment.evaluated_pid.Kd, 4)}\n\n` : ''}` +
            `性能评分: ${this.formatScore100(data.evaluation?.performance_score, 1)}/100\n` +
            `综合评分: ${this.formatScore100(data.evaluation?.final_rating, 1)}/100`;

          navigator.clipboard.writeText(text).then(() => {
            alert('参数已复制到剪贴板！');
          });
        },

        plantTypeLabel(value) {
          const labels = {
            distillation_column: '精馏塔',
            atmospheric_column: '常压塔',
            vacuum_column: '减压塔',
            side_draw_column: '侧线抽出塔',
            reboiler: '再沸器',
            condenser: '冷凝器',
            reflux_system: '回流系统'
          };
          return labels[value] || value || '-';
        },

        scenarioLabel(value) {
          const labels = {
            startup: '开车',
            shutdown: '停车',
            steady_operation: '稳态生产',
            load_change: '变负荷',
            product_switch: '切产品',
            feed_switch: '原料切换',
            tower_pressure_fluctuation: '塔压波动',
            reflux_fluctuation: '回流波动',
            steam_fluctuation: '蒸汽波动',
            analyzer_maintenance: '分析仪维护',
            low_load: '低负荷',
            full_load: '满负荷'
          };
          return labels[value] || value || '-';
        },

        controlObjectLabel(value) {
          const labels = {
            top_temperature: '塔顶温度',
            middle_temperature: '塔中温度',
            bottom_temperature: '塔釜温度',
            tower_pressure: '塔压',
            reflux_flow: '回流流量',
            steam_flow: '蒸汽流量',
            feed_flow: '进料流量',
            cooling_flow: '冷却流量',
            side_draw_flow: '侧线抽出流量',
            level: '液位',
            composition: '产品组成'
          };
          return labels[value] || value || '-';
        },

        modelTypeLabel(value) {
          const labels = {
            FOPDT: '一阶加纯滞后模型（FOPDT）',
            SOPDT: '二阶加纯滞后模型（SOPDT）',
            FO: '纯一阶模型（FO）',
            IPDT: '积分加纯滞后模型（IPDT）'
          };
          return labels[value] || value || '-';
        },

        strategyDisplayLabel(value) {
          const labels = {
            LAMBDA: 'LAMBDA 策略',
            IMC: 'IMC 策略',
            ZN: 'Ziegler-Nichols 策略',
            COHEN_COON: 'Cohen-Coon 策略',
            AUTO: '自动选择策略'
          };
          return labels[value] || value || '-';
        },

        reasonCodeLabel(code) {
          const labels = {
            best_r2: '拟合优度最高',
            stable_fit: '模型拟合稳定',
            low_rmse: '误差较低',
            balanced_response: '响应更均衡',
            robust_choice: '稳健性更优',
            experience_adjusted: '已结合历史经验修正',
            knowledge_constrained: '已应用知识约束'
          };
          return labels[code] || code || '-';
        },

        riskLevelLabel(value) {
          const labels = {
            low: '低风险',
            medium: '中风险',
            high: '高风险'
          };
          return labels[String(value || '').toLowerCase()] || value || '未标注';
        },

        reportStageLabel(value) {
          const normalized = String(value || '').trim();
          const labels = {
            data_analysis: '数据分析',
            identification: '系统辨识',
            system_identification: '系统辨识',
            knowledge: '知识增强',
            knowledge_enhancement: '知识增强',
            pid_tuning: 'PID整定',
            tuning: 'PID整定',
            evaluation: '评估反馈',
            '数据分析': '数据分析',
            '系统辨识': '系统辨识',
            '知识增强': '知识增强',
            'PID整定': 'PID整定',
            '评估反馈': '评估反馈'
          };
          return labels[normalized] || normalized || '-';
        },

        reportStageNarrative(stageLabel, stage) {
          const summary = this.truncateText(
            stage?.summary || stage?.output_summary || stage?.input_summary || '',
            180
          );
          const templates = {
            '数据分析': `已完成历史数据质量检查、候选窗口筛选与可用性判断。${summary || '当前未返回可展示的数据分析摘要。'}`,
            '系统辨识': `已完成候选模型辨识、拟合优度比较与模型收敛性检查。${summary || '当前未返回可展示的辨识摘要。'}`,
            '知识增强': `已结合经验中心与知识图谱约束对整定边界进行补充分析。${summary || '当前未返回可展示的知识增强摘要。'}`,
            'PID整定': `已基于入选模型生成 PID 参数建议，并形成可执行策略。${summary || '当前未返回可展示的整定摘要。'}`,
            '评估反馈': `已完成闭环性能评估，并给出上线动作与风险提示。${summary || '当前未返回可展示的评估摘要。'}`
          };
          return templates[stageLabel] || (summary || '暂无阶段摘要');
        },

        professionalRecommendationLabel(result) {
          if (!result) return '等待整定结果';
          if (result.evaluation?.passed) return '建议进入上线前确认，并在稳态工况下小步试投';
          return '建议继续回流优化，当前暂不推荐直接上线';
        },

        professionalReportStatusLabel(report) {
          const status = report?.report_meta?.status || '';
          const labels = {
            '已完成': '报告已完成',
            '待复核': '待复核后确认上线策略',
            '执行中': '整定执行中，报告内容持续更新',
            '未开始': '尚未生成整定报告'
          };
          return labels[status] || status || '状态未标注';
        },

        professionalReportStatusTone(report) {
          const status = report?.report_meta?.status || '';
          if (status === '已完成') return 'text-emerald-700 bg-emerald-50 border-emerald-200';
          if (status === '待复核') return 'text-amber-700 bg-amber-50 border-amber-200';
          if (status === '执行中') return 'text-blue-700 bg-blue-50 border-blue-200';
          return 'text-slate-700 bg-slate-50 border-slate-200';
        },

        professionalSummaryLine(report) {
          if (!report) return '当前暂无可生成的整定报告。';
          return `${report.task_context.loop_name || '当前回路'}在${report.task_context.scenario || '当前工况'}下完成整定分析，系统最终选定 ${this.modelTypeLabel(report.model_result.model_type)}，并生成 ${this.strategyDisplayLabel(report.pid_result.strategy)} 参数建议。综合评分为 ${this.formatScore100(report.evaluation_result.final_rating, 1)}/100，当前结论为：${report.evaluation_result.passed ? '建议进入上线前确认，并在受控条件下试投。' : '建议继续回流优化，暂不建议直接上线。'}`;
        },

        clearMessages() {
          this.messages = [];
          this.messageIdCounter = 0;
          localStorage.removeItem(this.messageStorageKey());
        },

        truncateText(text, maxLength = 120) {
          if (text == null || text === '') return '';
          const normalized = String(text).replace(/\s+/g, ' ').trim();
          return normalized.length > maxLength
            ? `${normalized.slice(0, maxLength - 1)}…`
            : normalized;
        },

        executionStageName(agentName, msg = null) {
          const name = String(agentName || '');
          const toolNames = Array.isArray(msg?.tools)
            ? msg.tools.map(tool => String(tool?.tool_name || '')).join(' ')
            : '';
          const haystack = `${name} ${toolNames}`;
          if (haystack.includes('history') || haystack.includes('load_data') || haystack.includes('fetch_history') || haystack.includes('\u6570\u636e')) return '\u6570\u636e\u5206\u6790';
          if (haystack.includes('fit_fopdt') || haystack.includes('identif') || haystack.includes('\u8fa8\u8bc6')) return '\u7cfb\u7edf\u8fa8\u8bc6';
          if (haystack.includes('knowledge') || haystack.includes('graph') || haystack.includes('\u672c\u4f53') || haystack.includes('\u77e5\u8bc6')) return '\u77e5\u8bc6\u589e\u5f3a';
          if (haystack.includes('tune_pid') || haystack.includes('PID')) return 'PID\u6574\u5b9a';
          if (haystack.includes('evaluate') || haystack.includes('\u8bc4\u4f30')) return '\u8bc4\u4f30\u53cd\u9988';
          return '\u6267\u884c\u6b65\u9aa4';
        },

        executionSummaryForMessage(msg) {
          if (!msg) return '\u5f53\u524d\u6b65\u9aa4\u6b63\u5728\u5904\u7406\u4e2d\u3002';
          const fromSummary = this.truncateText(msg.summary, 120);
          if (fromSummary) return fromSummary;
          const fromResponse = this.truncateText(msg.response || msg.content, 120);
          if (fromResponse) return fromResponse;
          const firstTool = Array.isArray(msg.tools) ? msg.tools[0] : null;
          const toolSummary = this.truncateText(firstTool?.summary || firstTool?.output_summary, 120);
          if (toolSummary) return toolSummary;
          return '\u5f53\u524d\u6b65\u9aa4\u6b63\u5728\u5904\u7406\u4e2d\u3002';
        },

        toolStatusLabel(tool) {
          if (tool?.is_error) return '\u5931\u8d25';
          if (tool?.result) return '\u5df2\u5b8c\u6210';
          return '\u6267\u884c\u4e2d';
        },

        toolStatusClass(tool) {
          if (tool?.is_error) return 'is-danger';
          if (tool?.result) return 'is-success';
          return 'is-warning';
        },

        summarizeTool(tool) {
          const summary = this.truncateText(tool?.summary || tool?.output_summary, 96);
          if (summary && summary !== '[object Object]') return summary;
          if (tool?.is_error) return '\u5de5\u5177\u6267\u884c\u5931\u8d25\uff0c\u8bf7\u5c55\u5f00\u67e5\u770b\u8fd4\u56de\u4fe1\u606f\u3002';
          if (tool?.result) return '\u5de5\u5177\u5df2\u8fd4\u56de\u7ed3\u679c\uff0c\u53ef\u5c55\u5f00\u67e5\u770b\u8be6\u60c5\u3002';
          return '\u5de5\u5177\u6b63\u5728\u6267\u884c\u4e2d\u3002';
        },

        parseToolPayload(payload) {
          if (payload == null) return null;
          if (typeof payload === 'object') return payload;
          const raw = String(payload).trim();
          if (!raw) return null;
          try {
            return JSON.parse(raw);
          } catch (error) {
            return null;
          }
        },

        summarizeValue(value, maxLength = 24) {
          if (value == null || value === '') return '-';
          if (Array.isArray(value)) {
            return this.truncateText(value.join('\u3001'), maxLength);
          }
          if (typeof value === 'object') {
            const keys = Object.keys(value).slice(0, 4);
            if (!keys.length) return '{}';
            return this.truncateText(keys.map(key => `${key}=${value[key]}`).join('\uff0c'), maxLength);
          }
          return this.truncateText(value, maxLength);
        },

        stringifyPayload(payload) {
          if (payload == null) return '';
          if (typeof payload === 'string') return payload;
          try {
            return JSON.stringify(payload);
          } catch (error) {
            return String(payload);
          }
        },

        toolInputSummary(tool) {
          const parsed = this.parseToolPayload(tool?.args);
          if (parsed && typeof parsed === 'object') {
            const keys = Object.keys(parsed).slice(0, 4);
            if (keys.length) {
              return keys.map(key => `${key}: ${this.summarizeValue(parsed[key], 24)}`).join(' | ');
            }
          }
          return this.truncateText(this.stringifyPayload(tool?.args), 96) || '\u65e0\u8f93\u5165\u53c2\u6570';
        },

        toolOutputSummary(tool) {
          const parsed = this.parseToolPayload(tool?.result);
          if (parsed && typeof parsed === 'object') {
            if (parsed.raw_content) return this.truncateText(parsed.raw_content, 120);
            if (parsed.summary) return this.truncateText(parsed.summary, 120);
            if (parsed.message) return this.truncateText(parsed.message, 120);
            if (parsed.detail) return this.truncateText(parsed.detail, 120);
            const keys = Object.keys(parsed).slice(0, 5);
            if (keys.length) return `\u8fd4\u56de\u5b57\u6bb5\uff1a${keys.join('\u3001')}`;
          }
          const rawText = this.stringifyPayload(tool?.result);
          if (rawText === '[object Object]') return '\u5df2\u8fd4\u56de\u7ed3\u6784\u5316\u7ed3\u679c\uff0c\u8bf7\u5c55\u5f00\u67e5\u770b\u8be6\u60c5\u3002';
          return this.truncateText(rawText, 120) || '\u6682\u65e0\u5de5\u5177\u7ed3\u679c';
        },

        /* fixTuningSidebarCopy() {
          const root = document.querySelector('#app');
          if (!root) return;
          const tuningPanel = root.querySelector('[v-if="currentPage === \\'tuning\\'"]') || root.querySelector('.p-6.space-y-4.flex-1');
          if (!tuningPanel) return;

          const labels = tuningPanel.querySelectorAll('label.block.text-sm.font-semibold.text-slate-700.mb-2');
          if (labels[1]) labels[1].textContent = '数据来源';
          if (labels[3]) labels[3].textContent = '上传 CSV 数据文件';
          if (labels[4]) labels[4].textContent = '回路类型';

          const historyPanel = tuningPanel.querySelector('.rounded-xl.border.border-slate-200.bg-slate-50.p-4.space-y-4');
          if (historyPanel) {
            const headerTitle = historyPanel.querySelector('.text-sm.font-semibold.text-slate-700');
            if (headerTitle) headerTitle.textContent = '历史数据参数';

            const note = historyPanel.querySelector('p.text-xs.text-amber-700');
            if (note) note.remove();

            const historyLabels = historyPanel.querySelectorAll('label.block.text-sm.font-semibold.text-slate-700.mb-2');
            if (historyLabels[1]) historyLabels[1].textContent = '开始时间';
            if (historyLabels[2]) historyLabels[2].textContent = '结束时间';

            const collapseButton = historyPanel.querySelector('button[type="button"]');
            if (collapseButton) {
              const spans = collapseButton.querySelectorAll('span');
              if (spans[0]) spans[0].textContent = this.historyPanelCollapsed ? '▼' : '▲';
              if (spans[1]) spans[1].remove();
            }
          }

          const sourceButtons = tuningPanel.querySelectorAll('.grid.grid-cols-2.gap-3 > button');
          if (sourceButtons[0]) {
            const texts = sourceButtons[0].querySelectorAll('div');
            if (texts[0]) texts[0].textContent = '上传 CSV';
            if (texts[1]) texts[1].textContent = '使用本地文件直接发起整定';
          }
          if (sourceButtons[1]) {
            const texts = sourceButtons[1].querySelectorAll('div');
            if (texts[0]) texts[0].textContent = '获取历史数据';
            if (texts[1]) texts[1].textContent = '按回路参数从历史系统取数';
          }

          const uploadBlock = tuningPanel.querySelector('input[type="file"][accept=".csv"]')?.parentElement;
          if (uploadBlock) {
            const uploadButton = uploadBlock.querySelector('button');
            if (uploadButton) uploadButton.innerHTML = '<span class="text-xl mr-2">📄</span> 选择 CSV 文件';
            const uploadedText = uploadBlock.querySelector('p.text-xs.text-slate-600');
            if (uploadedText && this.uploadedFile) uploadedText.textContent = `已选择：${this.uploadedFile.name}`;
          }
        } */

        professionalReportId() {
          const base = this.latestTuningResultMessage?.id || this.loopName || this.loopUri || 'pid';
          const safe = String(base).replace(/[^a-zA-Z0-9_-]+/g, '_');
          return `report_${safe}`;
        },

        reportStageSummary(stage) {
          if (!stage) return '-';
          const summary = Array.isArray(stage.key_findings) && stage.key_findings.length
            ? stage.key_findings.join('；')
            : (stage.output_summary || stage.input_summary || '-');
          return this.truncateText(summary, 160);
        },

        exportProfessionalReport(format = 'markdown') {
          const payload = this.professionalReportPayload;
          if (!payload) return;
          const slug = payload?.report_meta?.report_id || this.professionalReportId();
          if (format === 'json') {
            this.downloadTextFile(`${slug}.json`, JSON.stringify(payload, null, 2), 'application/json;charset=utf-8');
            return;
          }
          if (format === 'html') {
            this.downloadTextFile(`${slug}.html`, this.professionalReportHtml, 'text/html;charset=utf-8');
            return;
          }
          if (format === 'pdf') {
            const reportWindow = window.open('', '_blank', 'width=1200,height=900');
            if (!reportWindow) return;
            reportWindow.document.write(this.professionalReportHtml);
            reportWindow.document.close();
            reportWindow.focus();
            setTimeout(() => reportWindow.print(), 250);
            return;
          }
          this.downloadTextFile(`${slug}.md`, this.professionalReportMarkdown, 'text/markdown;charset=utf-8');
        },

        downloadTextFile(filename, content, mimeType = 'text/plain;charset=utf-8') {
          const blob = new Blob([content], { type: mimeType });
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = filename;
          document.body.appendChild(link);
          link.click();
          link.remove();
          URL.revokeObjectURL(url);
        }
      },
      computed: {
        latestTuningResultMessage() {
          return [...(this.messages || [])].reverse().find(msg => msg.type === 'result') || null;
        },
        latestTuningResultData() {
          return this.latestTuningResultMessage?.data || null;
        },
        professionalReportPayload() {
          const result = this.latestTuningResultData;
          const taskInput = this.taskInputMessage?.content || '';
          const stages = this.executionFlowGroups.map(group => {
            const stageLabel = this.reportStageLabel(group.stage);
            return {
              stage_id: group.stage,
              stage_name: stageLabel,
              agent_name: group.agent,
              status: group.status,
              duration_ms: 0,
              input_summary: group.tools?.[0] ? this.toolInputSummary(group.tools[0]) : '',
              output_summary: this.reportStageNarrative(stageLabel, group),
              key_findings: [group.summary || group.output_summary].filter(Boolean),
              tool_calls: (group.tools || []).map(tool => ({
                tool_name: tool.tool_name || 'tool',
                status: tool.success === false ? 'failed' : 'completed',
                input_summary: this.toolInputSummary(tool),
                output_summary: this.toolOutputSummary(tool)
              }))
            };
          });
          const loopTypeLabel = this.strategyLabLoopTypeLabel(this.loopType);
          const scenarioLabel = this.scenarioLabel(this.scenario);
          const plantTypeLabel = this.plantTypeLabel(this.plantType);
          const controlObjectLabel = this.controlObjectLabel(this.controlObject);
          const recommendation = this.professionalRecommendationLabel(result);
          return {
            report_meta: {
              report_id: this.professionalReportId(),
              task_id: this.latestTuningResultMessage?.id || '',
              title: 'PID智能整定专业报告',
              generated_at: new Date().toISOString(),
              status: result ? (result.evaluation?.passed ? '已完成' : '待复核') : (this.loading ? '执行中' : '未开始'),
              version: 'v1'
            },
            executive_summary: {
              summary_line: `${this.loopName || '当前回路'}在${scenarioLabel}下完成整定分析，最终选择${this.modelTypeLabel(result?.model?.modelType)}，并推荐${this.strategyDisplayLabel(result?.pidParams?.strategy || result?.tuningAdvice?.strategy)}。`,
              recommendation
            },
            task_context: {
              loop_name: this.loopName || '',
              loop_uri: this.loopUri || '',
              loop_type: loopTypeLabel,
              scenario: scenarioLabel,
              plant_type: plantTypeLabel,
              control_object: controlObjectLabel,
              data_source: this.dataSource === 'history' ? '历史接口' : 'CSV 上传',
              start_time: this.startTime || '',
              end_time: this.endTime || '',
              sampling_interval_sec: Number(this.historyWindow || 1),
              task_goal: '为当前控制回路生成 PID 整定建议',
              user_request: taskInput
            },
            data_summary: {
              point_count: result?.dataAnalysis?.points ?? 0,
              duration_sec: result?.dataAnalysis?.duration_sec ?? 0,
              valid_point_count: result?.dataAnalysis?.valid_points ?? 0,
              sampling_time_sec: result?.dataAnalysis?.sampling_time_sec ?? Number(this.historyWindow || 1),
              candidate_window_count: result?.dataAnalysis?.candidate_window_count ?? 0,
              selected_window_id: result?.dataAnalysis?.selected_window_id || '',
              selected_window_reason: result?.dataAnalysis?.selection_reason || '',
              step_detected: result?.dataAnalysis?.step_detected ?? true,
              noise_level: result?.dataAnalysis?.noise_level || '',
              quality_flags: result?.dataAnalysis?.quality_flags || [],
              data_risks: result?.dataAnalysis?.risks || []
            },
            agent_stages: stages,
            model_result: {
              model_type: result?.model?.modelType || '',
              model_params: {
                K: result?.model?.K ?? null,
                T: result?.model?.T ?? null,
                L: result?.model?.L ?? null,
                T1: result?.model?.T1 ?? null,
                T2: result?.model?.T2 ?? null,
                Ki_proc: result?.model?.Ki_proc ?? null
              },
              fit_metrics: {
                r2: result?.model?.fitMetrics?.r2 ?? null,
                rmse: result?.model?.fitMetrics?.rmse ?? null,
                normalized_rmse: result?.model?.fitMetrics?.normalized_rmse ?? null,
                confidence: result?.model?.confidence ?? null
              },
              selection_reason: result?.model?.modelSelectionReason || '',
              reason_codes: result?.model?.reasonCodes || []
            },
            pid_result: {
              strategy: result?.pidParams?.strategy || result?.tuningAdvice?.strategy || '',
              kp: result?.pidParams?.Kp ?? null,
              ki: result?.pidParams?.Ki ?? null,
              kd: result?.pidParams?.Kd ?? null,
              ti: result?.pidParams?.Ti ?? null,
              td: result?.pidParams?.Td ?? null,
              tuning_summary: result?.tuningAdvice?.summary || '',
              experience_adjusted: Boolean(result?.memory?.experience_used),
              knowledge_constrained: Boolean(result?.knowledge?.constraints?.length),
              supporting_rules: result?.knowledge?.constraints || []
            },
            evaluation_result: {
              passed: Boolean(result?.evaluation?.passed),
              final_rating: result?.evaluation?.final_rating ?? null,
              performance_score: result?.evaluation?.performance_score ?? null,
              method_confidence: result?.evaluation?.method_confidence ?? null,
              pass_threshold: result?.evaluation?.pass_threshold ?? null,
              failure_reason: result?.evaluation?.failure_reason || '',
              risk_level: result?.evaluation?.risk_level || '',
              highlights: result?.evaluation?.highlights || [],
              weaknesses: result?.evaluation?.weaknesses || []
            },
            deployment_advice: {
              recommended_action: result?.evaluation?.feedback_action || recommendation,
              go_live_strategy: result?.evaluation?.go_live_strategy || '',
              precheck_items: result?.evaluation?.precheck_items || [],
              postcheck_items: result?.evaluation?.postcheck_items || [],
              monitoring_metrics: result?.evaluation?.monitoring_metrics || [],
              rollback_advice: result?.evaluation?.rollback_advice || '',
              operator_notes: result?.evaluation?.operator_notes || ''
            },
            appendix: {
              raw_messages: this.messages || [],
              tool_call_records: stages.flatMap(item => item.tool_calls || []),
              candidate_models: result?.model?.candidate_models || [],
              candidate_windows: result?.dataAnalysis?.candidate_windows || [],
              raw_payloads: result || {}
            }
          };
        },
        professionalReportMarkdown() {
          const report = this.professionalReportPayload;
          if (!report) return '# PID智能整定专业报告\n\n暂无可导出的整定结果。';
          const stageLines = report.agent_stages.map((stage, idx) => {
            const tools = (stage.tool_calls || []).map(tool => `  - ${tool.tool_name}: ${tool.output_summary || '-'}`).join('\n');
            return `### ${idx + 1}. ${stage.stage_name}\n- 智能体：${stage.agent_name}\n- 状态：${stage.status}\n- 阶段结论：${stage.output_summary || '-'}\n${tools}`;
          }).join('\n\n');
          return [
            '# PID智能整定专业报告',
            `- 报告编号：${report.report_meta.report_id}`,
            `- 生成时间：${report.report_meta.generated_at}`,
            `- 回路名称：${report.task_context.loop_name || '-'}`,
            `- 数据来源：${report.task_context.data_source || '-'}`,
            '',
            '## 执行摘要',
            `- 摘要结论：${report.executive_summary.summary_line || '-'}`,
            `- 最终模型：${this.modelTypeLabel(report.model_result.model_type)}`,
            `- 推荐策略：${this.strategyDisplayLabel(report.pid_result.strategy)}`,
            `- 综合评分：${report.evaluation_result.final_rating ?? '-'}`,
            `- 评估结论：${report.evaluation_result.passed ? '通过' : '待复核'}`,
            `- 推荐动作：${report.executive_summary.recommendation || '-'}`,
            '',
            '## 任务背景',
            `- 用户请求：${report.task_context.user_request || '-'}`,
            `- 回路类型：${report.task_context.loop_type || '-'}`,
            `- 工况：${report.task_context.scenario || '-'}`,
            `- 装置类型：${report.task_context.plant_type || '-'}`,
            `- 控制对象：${report.task_context.control_object || '-'}`,
            `- 采样间隔：${report.task_context.sampling_interval_sec || '-'} 秒`,
            '',
            '## 数据概况',
            `- 数据点数：${report.data_summary.point_count ?? 0}`,
            `- 有效点数：${report.data_summary.valid_point_count ?? 0}`,
            `- 候选窗口数：${report.data_summary.candidate_window_count ?? 0}`,
            `- 选中窗口：${report.data_summary.selected_window_id || '-'}`,
            `- 选窗理由：${report.data_summary.selected_window_reason || '-'}`,
            '',
            '## 多智能体分析过程',
            stageLines,
            '',
            '## 最终整定结果',
            `- 模型：${this.modelTypeLabel(report.model_result.model_type)}`,
            `- Kp / Ki / Kd：${report.pid_result.kp ?? '-'} / ${report.pid_result.ki ?? '-'} / ${report.pid_result.kd ?? '-'}`,
            `- 选模理由：${report.model_result.selection_reason || '-'}`,
            `- PID 摘要：${report.pid_result.tuning_summary || '-'}`,
            '',
            '## 上线建议',
            `- 推荐动作：${report.deployment_advice.recommended_action || '-'}`,
            `- 上线方式：${report.deployment_advice.go_live_strategy || '-'}`,
            `- 回滚建议：${report.deployment_advice.rollback_advice || '-'}`,
            '',
            '## 风险提示',
            `- 风险等级：${this.riskLevelLabel(report.evaluation_result.risk_level)}`,
            `- 主要短板：${report.evaluation_result.failure_reason || report.evaluation_result.weaknesses?.join('；') || '暂无明显风险'}`
          ].join('\n');
        },
        professionalReportHtml() {
          const report = this.professionalReportPayload;
          const stageCards = (report.agent_stages || []).map(stage => `
            <section class="stage-card">
              <div class="stage-head">
                <div class="stage-tag">${stage.stage_name}</div>
                <div class="stage-status">${stage.status === 'completed' ? '已完成' : '处理中'}</div>
              </div>
              <div class="stage-title">${stage.agent_name}</div>
              <div class="stage-body">${stage.output_summary || '-'}</div>
            </section>
          `).join('');
          return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>PID智能整定专业报告</title>
  <style>
    @page { size: A4; margin: 16mm 14mm; }
    * { box-sizing: border-box; }
    body{
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
      background:#f8fafc;color:#0f172a;margin:0;line-height:1.7;
      -webkit-print-color-adjust: exact; print-color-adjust: exact;
    }
    .wrap{max-width:1040px;margin:0 auto;padding:28px 20px 40px}
    .report-header{
      display:flex;justify-content:space-between;align-items:flex-start;gap:24px;
      background:linear-gradient(135deg,#eaf2ff 0%,#f8fbff 100%);
      border:1px solid #dbe7ff;border-radius:24px;padding:28px 30px;margin-bottom:20px;
      page-break-inside:avoid;
    }
    .report-badge{
      display:inline-flex;align-items:center;gap:8px;padding:8px 14px;border-radius:999px;
      border:1px solid #c7d8ff;background:#fff;color:#2457d6;font-size:12px;font-weight:700;
      text-transform:uppercase;letter-spacing:.12em;
    }
    .report-meta{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}
    .meta-chip{border:1px solid #dbe4f0;background:#fff;border-radius:999px;padding:7px 12px;font-size:12px;color:#475569;}
    .headline{margin-top:18px;padding:16px 18px;border-radius:18px;background:#ffffff;border:1px solid #d9e7ff;color:#1e293b;font-size:15px;}
    .status-line{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
    .status-pill,.advice-pill{border-radius:999px;padding:8px 14px;font-size:13px;font-weight:600;}
    .status-pill{border:1px solid #dbe4f0;background:#fff;color:#334155;}
    .advice-pill{border:1px solid #dbe7ff;background:#eef4ff;color:#2457d6;}
    .section{background:#fff;border:1px solid #e2e8f0;border-radius:20px;padding:22px 24px;margin-bottom:18px;page-break-inside:avoid;}
    .section-title{display:flex;align-items:center;gap:10px;margin:0 0 14px;font-size:20px;font-weight:700;color:#0f172a;}
    .section-kicker{display:inline-block;margin-bottom:10px;font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.14em;}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
    .grid-3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
    .kpi{border:1px solid #e5e7eb;border-radius:16px;padding:16px;background:#fbfdff;page-break-inside:avoid;}
    .kpi-label{color:#64748b;font-size:12px;font-weight:600}
    .kpi-value{margin-top:6px;color:#0f172a;font-size:18px;font-weight:700}
    .muted{color:#64748b;font-size:13px}
    .body-copy{color:#334155;font-size:14px}
    h1{font-size:30px;line-height:1.25;margin:12px 0 8px}
    h2{font-size:22px;line-height:1.35;margin:0 0 12px}
    .summary-box,.note-box{margin-top:14px;padding:15px 16px;border-radius:16px;border:1px solid #e5e7eb;background:#f8fafc;color:#334155;}
    .stage-card{border:1px solid #e5e7eb;border-radius:16px;padding:16px;margin-bottom:14px;background:#fafcff;page-break-inside:avoid;}
    .stage-head{display:flex;justify-content:space-between;align-items:center;gap:12px}
    .stage-tag{font-size:12px;color:#2457d6;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
    .stage-status{font-size:12px;color:#475569;border:1px solid #dbe4f0;background:#fff;border-radius:999px;padding:5px 10px}
    .stage-title{font-size:18px;font-weight:700;color:#0f172a;margin-top:8px}
    .stage-body{margin-top:10px;color:#334155;font-size:14px}
    .result-highlight{border:1px solid #d9e7ff;background:#eef4ff;border-radius:18px;padding:16px 18px;color:#1e3a8a;font-size:14px;}
    @media print {
      .wrap{padding:0}
      .report-header,.section,.kpi,.stage-card{break-inside:avoid;page-break-inside:avoid}
    }
  </style>
</head>
<body><div class="wrap">
  <section class="report-header">
    <div>
      <div class="report-badge">PID智能整定报告</div>
      <h1>PID智能整定专业报告</h1>
      <div class="muted">适用于控制工程师复核、上线评审与报告归档。</div>
      <div class="report-meta">
        <span class="meta-chip">报告编号：${report.report_meta.report_id}</span>
        <span class="meta-chip">任务编号：${report.report_meta.task_id || '-'}</span>
        <span class="meta-chip">生成时间：${report.report_meta.generated_at}</span>
      </div>
      <div class="headline">${report.executive_summary.summary_line || '-'}</div>
      <div class="status-line">
        <span class="status-pill">${this.professionalReportStatusLabel(report)}</span>
        <span class="advice-pill">推荐动作：${report.executive_summary.recommendation || '-'}</span>
      </div>
    </div>
    <div style="min-width:220px">
      <div class="section-kicker">执行摘要</div>
      <div class="kpi">
        <div class="kpi-label">回路名称</div>
        <div class="kpi-value">${report.task_context.loop_name || '-'}</div>
      </div>
      <div class="kpi" style="margin-top:12px">
        <div class="kpi-label">最终模型</div>
        <div class="kpi-value">${this.modelTypeLabel(report.model_result.model_type)}</div>
      </div>
      <div class="kpi" style="margin-top:12px">
        <div class="kpi-label">综合评分</div>
        <div class="kpi-value">${report.evaluation_result.final_rating ?? '-'}</div>
      </div>
    </div>
  </section>
  <section class="section">
    <div class="section-kicker">任务背景</div>
    <div class="section-title">任务背景与数据概况</div>
    <div class="grid">
      <div class="kpi"><div class="kpi-label">回路类型</div><div class="kpi-value">${report.task_context.loop_type || '-'}</div></div>
      <div class="kpi"><div class="kpi-label">工况</div><div class="kpi-value">${report.task_context.scenario || '-'}</div></div>
      <div class="kpi"><div class="kpi-label">装置类型</div><div class="kpi-value">${report.task_context.plant_type || '-'}</div></div>
      <div class="kpi"><div class="kpi-label">控制对象</div><div class="kpi-value">${report.task_context.control_object || '-'}</div></div>
      <div class="kpi"><div class="kpi-label">数据来源</div><div class="kpi-value">${report.task_context.data_source || '-'}</div></div>
      <div class="kpi"><div class="kpi-label">采样间隔</div><div class="kpi-value">${report.task_context.sampling_interval_sec || '-'} 秒</div></div>
    </div>
    <div class="note-box">
      <div class="kpi-label">任务目标</div>
      <div class="body-copy" style="margin-top:6px;">${report.task_context.user_request || report.task_context.task_goal || '暂无任务描述'}</div>
    </div>
    <div class="grid" style="margin-top:16px">
      <div class="kpi"><div class="kpi-label">数据点数</div><div class="kpi-value">${report.data_summary.point_count ?? 0}</div></div>
      <div class="kpi"><div class="kpi-label">有效点数</div><div class="kpi-value">${report.data_summary.valid_point_count ?? 0}</div></div>
      <div class="kpi"><div class="kpi-label">候选窗口数</div><div class="kpi-value">${report.data_summary.candidate_window_count ?? 0}</div></div>
      <div class="kpi"><div class="kpi-label">选中窗口</div><div class="kpi-value">${report.data_summary.selected_window_id || '-'}</div></div>
    </div>
    <div class="summary-box">
      <div class="kpi-label">窗口选择理由</div>
      <div class="body-copy" style="margin-top:6px;">${report.data_summary.selected_window_reason || '暂无窗口说明'}</div>
    </div>
  </section>
  <section class="section">
    <div class="section-kicker">过程追溯</div>
    <div class="section-title">多智能体分析过程</div>
    ${stageCards}
  </section>
  <section class="section">
    <div class="section-kicker">最终结果</div>
    <div class="section-title">模型、参数与评估结论</div>
    <div class="grid-3">
      <div class="kpi">
        <div class="kpi-label">模型类型</div>
        <div class="kpi-value">${this.modelTypeLabel(report.model_result.model_type)}</div>
        <div class="body-copy" style="margin-top:10px;">K / T / L：${report.model_result.model_params.K ?? '-'} / ${report.model_result.model_params.T ?? '-'} / ${report.model_result.model_params.L ?? '-'}</div>
        <div class="body-copy" style="margin-top:6px;">R²：${report.model_result.fit_metrics.r2 ?? '-'}，置信度：${report.model_result.fit_metrics.confidence ?? '-'}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">推荐策略</div>
        <div class="kpi-value">${this.strategyDisplayLabel(report.pid_result.strategy)}</div>
        <div class="body-copy" style="margin-top:10px;">Kp / Ki / Kd：${report.pid_result.kp ?? '-'} / ${report.pid_result.ki ?? '-'} / ${report.pid_result.kd ?? '-'}</div>
        <div class="body-copy" style="margin-top:6px;">Ti / Td：${report.pid_result.ti ?? '-'} / ${report.pid_result.td ?? '-'}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">评估结论</div>
        <div class="kpi-value">${report.evaluation_result.passed ? '通过' : '待复核'}</div>
        <div class="body-copy" style="margin-top:10px;">综合评分：${report.evaluation_result.final_rating ?? '-'}</div>
        <div class="body-copy" style="margin-top:6px;">风险等级：${this.riskLevelLabel(report.evaluation_result.risk_level)}</div>
      </div>
    </div>
    <div class="summary-box">
      <div class="kpi-label">选模理由</div>
      <div class="body-copy" style="margin-top:6px;">${report.model_result.selection_reason || '暂无选模理由'}</div>
    </div>
    <div class="summary-box">
      <div class="kpi-label">PID 摘要</div>
      <div class="body-copy" style="margin-top:6px;">${report.pid_result.tuning_summary || '暂无整定摘要'}</div>
    </div>
  </section>
  <section class="section">
    <div class="section-kicker">上线建议</div>
    <div class="section-title">上线建议与风险提示</div>
    <div class="grid">
      <div class="result-highlight">
        <strong>推荐动作：</strong>${report.deployment_advice.recommended_action || '-'}<br />
        <strong>上线方式：</strong>${report.deployment_advice.go_live_strategy || '-'}
      </div>
      <div class="result-highlight" style="background:#fff7ed;border-color:#fed7aa;color:#9a3412;">
        <strong>回滚建议：</strong>${report.deployment_advice.rollback_advice || '-'}<br />
        <strong>风险等级：</strong>${this.riskLevelLabel(report.evaluation_result.risk_level)}
      </div>
    </div>
  </section>
</div></body></html>`;
        },
        shellSecondaryItems() {
          return this.shellSecondaryItemsFor(this.currentPage);
        },
        shellPageMeta() {
          const map = {
            tuning: {
              title: '智能整定',
              description: '围绕当前回路组织参数输入、执行过程与调参结果。'
            },
            experience: {
              title: '经验中心',
              description: '查看历史整定经验，检索相似回路并复用知识。'
            },
            'case-library': {
              title: '案例库',
              description: '按场景和失败模式管理案例资产，支持对照分析。'
            },
            'strategy-lab': {
              title: '策略案例',
              description: '管理候选策略、评测结果和演化版本。'
            },
            'system-config': {
              title: '系统配置',
              description: '统一管理模型、历史数据与知识图谱服务接入。'
            }
          };
          return map[this.currentPage] || { title: '工作台', description: '' };
        },
        shellStatusCards() {
          if (this.currentPage === 'tuning') {

            return [
              {
                title: '当前阶段',
                value: this.executionOverview.currentStage || '等待启动',
                helper: this.executionOverview.currentAgent || '尚未开始'
              },
              {
                title: '数据来源',
                value: this.dataSource === 'history' ? '历史接口' : 'CSV 上传',
                helper: `采样间隔 ${this.historyWindow || 1}s`
              },
              {
                title: '当前模型',
                value: this.latestTuningResultData?.model?.modelType || (this.loading ? '识别中' : '待生成'),
                helper: this.latestTuningResultData?.model?.modelSelectionReason || '结果生成后会在这里显示模型选择依据'
              },
              {
                title: '结果评分',
                value: this.latestTuningResultData ? `${this.formatScore100(this.latestTuningResultData.evaluation?.final_rating, 1)}/100` : (this.loading ? '处理中' : '暂无'),
                helper: this.latestTuningResultData?.evaluation?.passed ? '当前结果已通过评估' : (this.latestTuningResultData?.evaluation?.failure_reason || '用于跟踪多智能体执行过程')
              }
            ];

          }
          if (this.currentPage === 'experience') {
            return [
              {
                title: '经验总数',
                value: String(this.experienceStats.total_count ?? 0),
                helper: '当前已沉淀的经验条目'
              },
              {
                title: '通过经验',
                value: String(this.experienceStats.passed_count ?? 0),
                helper: '可优先复用的经验数量'
              },
              {
                title: '相似检索',
                value: this.experienceSearchLoading ? '检索中' : '待检索',
                helper: '支持按回路和模型参数检索'
              }
            ];
          }
          if (this.currentPage === 'strategy-lab') {
            return [
              {
                title: '候选数量',
                value: String(this.strategyLabCandidates.length),
                helper: '当前实验室候选策略总数'
              },
              {
                title: '基准案例',
                value: String(this.strategyLabCases.length),
                helper: '用于离线评测的案例集合'
              },
              {
                title: '当前选中',
                value: this.selectedStrategyLabCandidate ? this.strategyLabCandidateLabel(this.selectedStrategyLabCandidate.id) : '未选择',
                helper: '可进一步查看 Prompt、报告和源码'
              }
            ];
          }
          if (this.currentPage === 'system-config') {
            return [
              {
                title: '模型名称',
                value: this.systemConfig?.model?.name || '未配置',
                helper: '当前整定任务默认使用的模型'
              },
              {
                title: '历史数据服务',
                value: this.systemConfig?.integration?.history_data_api_url ? '已配置' : '未配置',
                helper: this.systemConfig?.integration?.history_data_api_url || '请先配置历史数据接口'
              },
              {
                title: '知识图谱服务',
                value: this.systemConfig?.integration?.knowledge_graph_api_url ? '已配置' : '未配置',
                helper: this.systemConfig?.integration?.knowledge_graph_api_url || '请先配置知识图谱接口'
              }
            ];
          }
          return [
            {
              title: '案例数量',
              value: String(this.caseLibraryStats.total_count ?? 0),
              helper: '当前案例库的总案例数'
            },
            {
              title: '已选案例',
              value: this.selectedCase?.title || '未选择',
              helper: '可继续查看案例详情和上下文'
            }
          ];
        },
        selectedStrategyLabCandidate() {
          return this.strategyLabCandidates.find(item => item.id === this.selectedStrategyLabCandidateId) || null;
        },
        comparedStrategyLabCandidate() {
          return this.strategyLabCandidates.find(item => item.id === this.strategyLabCompareCandidateId) || null;
        },
        taskInputMessage() {
          return this.messages.find(msg => msg.type === 'user') || null;
        },
        executionFlowGroups() {
          const agentTurns = (this.messages || []).filter(msg => msg.type === 'agent_turn');
          const activeIndex = this.loading && agentTurns.length ? agentTurns.length - 1 : -1;
          return agentTurns.map((msg, index) => ({
            id: msg.id,
            agent: msg.agent || '\u667a\u80fd\u4f53',
            stage: this.executionStageName(msg.agent, msg),
            status: index === activeIndex ? 'running' : 'completed',
            statusLabel: index === activeIndex ? '\u6267\u884c\u4e2d' : '\u5df2\u5b8c\u6210',
            collapsed: Boolean(msg.collapsed),
            summary: this.executionSummaryForMessage(msg),
            response: msg.response || msg.content || '',
            tools: Array.isArray(msg.tools) ? msg.tools : [],
            toolCount: Array.isArray(msg.tools) ? msg.tools.length : 0
          }));
        },
        executionVisibleGroups() {
          const groups = this.executionFlowGroups;
          if (this.executionViewMode === 'key-results') {
            return groups.map(group => ({
              ...group,
              tools: [],
              toolCount: group.toolCount
            }));
          }
          return groups;
        },
        executionStageEntries() {
          const stageNames = this.progressSteps.map((step, idx) => ({
            name: this.progressStepLabel(step, idx),
            icon: this.progressStepIcon(step, idx)
          }));
          const stageOrder = stageNames.map(item => item.name);
          const touched = this.executionFlowGroups.map(group => group.stage);
          const highestIndex = touched.length ? Math.max(...touched.map(name => stageOrder.indexOf(name)).filter(index => index >= 0)) : -1;
          const currentStage = this.loading ? this.executionOverview.currentStage : '';
          return stageNames.map((item, idx) => {
            let state = 'pending';
            if (idx < highestIndex) state = 'completed';
            if (idx === highestIndex && highestIndex >= 0 && !this.loading) state = 'completed';
            if (item.name === currentStage) state = 'running';
            if (this.latestTuningResultData?.evaluation?.passed && idx <= Math.max(highestIndex, stageOrder.length - 1)) {
              state = 'completed';
            }
            return {
              ...item,
              state,
              statusLabel: state === 'running' ? '\u6b63\u5728\u6267\u884c' : state === 'completed' ? '\u5df2\u5b8c\u6210' : '\u7b49\u5f85\u4e2d'
            };
          });
        },
        executionOverview() {
          const groups = this.executionFlowGroups;
          const totalTools = groups.reduce((sum, group) => sum + group.toolCount, 0);
          const running = groups.find(group => group.status === 'running') || null;
          const latestResult = this.latestTuningResultData;
          const defaultInsight = '\u53d1\u8d77\u6574\u5b9a\u540e\uff0c\u8fd9\u91cc\u4f1a\u6301\u7eed\u66f4\u65b0\u9636\u6bb5\u6458\u8981\u3002';
          const resultInsight = latestResult?.evaluation?.passed
            ? '\u5f53\u524d\u7ed3\u679c\u5df2\u6ee1\u8db3\u8bc4\u4f30\u9608\u503c\uff0c\u53ef\u5207\u6362\u5230\u7ed3\u679c\u9875\u7ee7\u7eed\u67e5\u770b\u53c2\u6570\u548c\u4e0a\u7ebf\u5efa\u8bae\u3002'
            : (latestResult?.evaluation?.failure_reason || running?.summary || defaultInsight);
          return {
            totalMessages: (this.messages || []).length,
            totalAgents: groups.length,
            totalTools,
            completedAgents: groups.filter(group => group.status === 'completed').length,
            currentStage: running?.stage || (groups.length ? groups[groups.length - 1].stage : '\u7b49\u5f85\u542f\u52a8'),
            currentAgent: running?.agent || (groups.length ? groups[groups.length - 1].agent : '\u5c1a\u672a\u5f00\u59cb'),
            finalState: latestResult ? (latestResult.evaluation?.passed ? '\u5df2\u901a\u8fc7\u8bc4\u4f30' : '\u5f85\u7ee7\u7eed\u4f18\u5316') : (this.loading ? '\u6267\u884c\u4e2d' : '\u672a\u5f00\u59cb'),
            finalStateClass: latestResult ? (latestResult.evaluation?.passed ? 'is-success' : 'is-warning') : (this.loading ? 'is-warning' : ''),
            insight: resultInsight
          };
        }
      },
      watch: {
        selectedStrategyLabCandidateId() {
          this.saveStrategyLabState();
        },
        strategyLabCompareCandidateId() {
          this.saveStrategyLabState();
        }
      },
      mounted() {
        this.loadMessages();
        this.loadStrategyLabState();
        this.shellSection = this.shellSecondaryItemsFor(this.currentPage)[0]?.id || '';
        this.bindShellSecondaryFallback();
      },
      beforeUnmount() {
      }
    }).mount('#app');
