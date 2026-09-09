/** What this build is, what it can do, and the two ways to start over. */

import { useQueryClient } from '@tanstack/react-query'
import { Compass, RotateCcw, Sparkles, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import { Badge, Button, Callout, Disclosure, Spinner } from '@/components/ui/primitives'
import { ApiError, api } from '@/lib/api'
import { titleCase } from '@/lib/format'
import { useCapabilities } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { Group, SectionHead } from '../controls'
import { useSettingsController } from '../state'

export function AboutSection() {
  const capabilities = useCapabilities()
  const client = useQueryClient()
  const notify = useApp((s) => s.notify)
  const startTour = useApp((s) => s.startTour)
  const closeSettings = useApp((s) => s.closeSettings)
  const setAppearance = useApp((s) => s.setAppearance)
  const { config, updateRoot } = useSettingsController()

  const [confirming, setConfirming] = useState(false)
  const [resetting, setResetting] = useState(false)

  const info = capabilities.data

  const resetAll = async () => {
    setResetting(true)
    try {
      const result = await api.settings.reset()
      setAppearance({
        theme: result.config.appearance.theme,
        accent: result.config.appearance.accent,
        density: result.config.appearance.density,
        fontScale: result.config.appearance.font_scale,
        reducedMotion: result.config.appearance.reduced_motion,
        numericLocale: result.config.appearance.numeric_locale,
      })
      await client.invalidateQueries()
      setConfirming(false)
      notify({
        tone: 'positive',
        title: 'Settings reset',
        body: 'Every section is back to the shipped defaults. Your API key and system prompts were not touched.',
      })
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'Settings were not reset',
        body: error instanceof ApiError ? error.message : 'The dashboard service rejected the request.',
      })
    } finally {
      setResetting(false)
    }
  }

  return (
    <>
      <SectionHead title="About" description="This build of the engine, and what it is made of." />

      {!info && capabilities.isLoading && (
        <div className="row gap-2">
          <Spinner /> <span className="t-small muted">Reading the engine capabilities</span>
        </div>
      )}

      {info && (
        <>
          <div className="settings__switch" style={{ alignItems: 'center' }}>
            <span
              className="row center"
              style={{
                width: 38,
                height: 38,
                flex: 'none',
                borderRadius: 'var(--r-md)',
                background: 'var(--accent-50)',
                color: 'var(--accent)',
              }}
            >
              <Compass size={19} />
            </span>
            <span className="settings__switch-text">
              <span className="t-body semibold">{info.name}</span>
              <span className="t-tiny muted">{info.full_name}</span>
            </span>
            <Badge tone="accent" mono>
              v{info.version}
            </Badge>
          </div>

          <dl className="kv">
            <dt>Version</dt>
            <dd className="tabular">{info.version}</dd>
            <dt>Python</dt>
            <dd className="tabular">{info.python}</dd>
            <dt>Platform</dt>
            <dd>{info.platform}</dd>
            <dt>Config version</dt>
            <dd className="tabular">{config.version}</dd>
          </dl>

          <Group title={`Pipeline stages (${info.stages.length})`}>
            <div className="row gap-2 wrap">
              {info.stages.map((stage, index) => (
                <Badge key={stage} outline>
                  {index + 1}. {stage}
                </Badge>
              ))}
            </div>
          </Group>

          <Disclosure title="Agents" subtitle={`${info.agents.length} roles in the loop`}>
            <div className="stack gap-3">
              {info.agents.map((agent) => (
                <div key={agent.role} className="stack" style={{ gap: 1 }}>
                  <span className="row gap-2">
                    <span className="t-small semibold">{agent.label}</span>
                    <Badge>stage {agent.stage}</Badge>
                  </span>
                  <span className="t-tiny muted">{agent.summary}</span>
                </div>
              ))}
            </div>
          </Disclosure>

          <Disclosure title="Tools" subtitle={`${info.tools.length} analysis tools available to a plan`}>
            <div className="stack gap-3">
              {info.tools.map((tool) => (
                <div key={tool.name} className="stack" style={{ gap: 1 }}>
                  <span className="row gap-2">
                    <span className="t-small semibold">{tool.name}</span>
                    <Badge outline>{titleCase(tool.family)}</Badge>
                  </span>
                  {tool.summary && <span className="t-tiny muted">{tool.summary}</span>}
                </div>
              ))}
            </div>
          </Disclosure>
        </>
      )}

      <Group title="Starting over">
        <div className="settings__switch">
          <span className="settings__switch-text">
            <span className="t-small semibold">Guided tour</span>
            <span className="t-tiny muted">Walks through the interface from the top, one step at a time.</span>
          </span>
          <Button
            icon={<Sparkles size={13} />}
            onClick={() => {
              void updateRoot({ tour_completed: [] })
              startTour()
              closeSettings()
            }}
          >
            Restart the tour
          </Button>
        </div>

        <div className="settings__switch">
          <span className="settings__switch-text">
            <span className="t-small semibold">First-run setup</span>
            <span className="t-tiny muted">
              {config.onboarding_complete
                ? 'Marked complete. Show it again the next time the dashboard loads.'
                : 'Not yet completed; it will appear on the next load.'}
            </span>
          </span>
          <Button
            disabled={!config.onboarding_complete}
            onClick={() => void updateRoot({ onboarding_complete: false })}
          >
            Show it again
          </Button>
        </div>
      </Group>

      <Group title="Reset">
        {confirming ? (
          <Callout
            tone="critical"
            icon={<TriangleAlert size={15} />}
            title="Reset every setting to its default?"
            action={
              <div className="row gap-2">
                <Button size="sm" onClick={() => setConfirming(false)}>
                  Cancel
                </Button>
                <Button variant="danger" size="sm" loading={resetting} onClick={() => void resetAll()}>
                  Reset everything
                </Button>
              </div>
            }
          >
            Connection, models, engine, budgets, cost guardrails, local backend, batch, data roots, appearance, and
            instructions all go back to the shipped values. Your stored API key and any edited system prompts are left
            alone.
          </Callout>
        ) : (
          <div className="settings__switch">
            <span className="settings__switch-text">
              <span className="t-small semibold">Reset all settings</span>
              <span className="t-tiny muted">Restores every section to the shipped defaults. This cannot be undone.</span>
            </span>
            <Button variant="danger" icon={<RotateCcw size={13} />} onClick={() => setConfirming(true)}>
              Reset
            </Button>
          </div>
        )}
      </Group>
    </>
  )
}
