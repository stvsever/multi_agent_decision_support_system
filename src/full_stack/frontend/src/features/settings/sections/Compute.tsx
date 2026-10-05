/**
 * Where inference runs, which model answers, and the hardware behind it.
 *
 * The three questions used to sit in three sections and could contradict each
 * other. They are one screen now: the backend choice decides which credential,
 * which model chooser, and which runtime detail is shown.
 */

import { ExternalLink, PlugZap, TriangleAlert, Undo2 } from 'lucide-react'
import { Fragment, useEffect, useMemo, useState } from 'react'
import { ModelPicker } from '@/components/ui/ModelPicker'
import { Badge, Button, Callout, Disclosure, Field, Segmented, Tooltip } from '@/components/ui/primitives'
import { isDecisionModelId, useDecisionModels } from '@/lib/decision'
import { compactNumber, tokens, usdPerMillion } from '@/lib/format'
import { useCapabilities, useCatalog, useConnectivity, useSettings } from '@/lib/hooks'
import { AGENT_ROLES, type AgentRole, type BackendName, type ReasoningEffort, type RoleMap } from '@/lib/types'
import { Grid, Group, NumberControl, NumberSetting, SectionHead, SliderControl, TextSetting } from '../controls'
import { HuggingFaceKeyPanel, OpenRouterKeyPanel } from '../credentials'
import { CompanionPanel, DecisionModelSettings, type CompanionRequest } from '../decision'
import { DeploymentPlan, HfEmbeddingSearch, HfModelSearch, MachineFacts, SelfHostedRuntime } from '../deploy'
import { useSettingsController } from '../state'

const BACKENDS: { value: BackendName; label: string; blurb: string }[] = [
  {
    value: 'openrouter',
    label: 'OpenRouter',
    blurb:
      'One key reaches every hosted model, with live prices the cost estimate can use. This is the path the engine is tuned for.',
  },
  {
    value: 'local',
    label: 'Self-hosted',
    blurb:
      'Weights run on hardware you control, here or on a cluster you submit to. Nothing leaves the host, there is no per-token cost, and plan steps run one at a time.',
  },
]

const EFFORTS: { value: ReasoningEffort; label: string }[] = [
  { value: 'provider_default', label: 'Provider default' },
  { value: 'off', label: 'Off' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
]

const ROLE_DEFAULTS: {
  model: RoleMap<string>
  maxTokens: RoleMap<number>
  contextWindow: RoleMap<number>
  temperature: RoleMap<number>
} = {
  model: { orchestrator: '', integrator: '', predictor: '', critic: '', communicator: '', tool: '' },
  contextWindow: { orchestrator: 0, integrator: 0, predictor: 0, critic: 0, communicator: 0, tool: 0 },
  maxTokens: { orchestrator: 8000, integrator: 8000, predictor: 12000, critic: 6000, communicator: 16000, tool: 4000 },
  temperature: { orchestrator: 0.3, integrator: 0.3, predictor: 0.2, critic: 0.2, communicator: 0.2, tool: 0.5 },
}

/** Where the engine took a role's context window from, in the words of the settings screen. */
const WINDOW_SOURCES: Record<string, string> = {
  decision_state: 'decision state limit',
  role_override: 'value set here',
  local_served: 'served length',
  catalog: 'catalog',
  known_table: 'built-in table',
  configured: 'context window above',
  fallback: 'generic fallback',
}

interface ResolvedWindow {
  tokens: number
  source: string
}

const ROLE_FALLBACK: Record<AgentRole, string> = {
  orchestrator: 'Reads the coverage map and writes the plan of tool steps.',
  integrator: 'Fuses the step outputs into one evidence representation.',
  predictor: 'Produces the phenotype outputs and the evidence chain.',
  critic: 'Scores the prediction and decides whether to iterate.',
  communicator: 'Writes the deep phenotype report.',
  tool: 'Runs the individual analysis tools that make up each plan step.',
}

export function ComputeSection() {
  const { config, update } = useSettingsController()
  const capabilities = useCapabilities()
  const connectivity = useConnectivity()
  const { data: saved } = useSettings()
  const resolvedWindows = (saved?.effective?.context_windows ?? {}) as Partial<Record<AgentRole, ResolvedWindow>>

  const backend = config.connection.backend
  const models = config.models
  const local = config.local
  const isLocal = backend === 'local'

  const links = capabilities.data?.provider_links ?? {}
  const catalog = useCatalog({ embedding: false, limit: 500 }, !isLocal)
  const row = useMemo(
    () => catalog.data?.models.find((entry) => entry.id === models.default_model),
    [catalog.data, models.default_model],
  )

  const active = BACKENDS.find((option) => option.value === backend) ?? BACKENDS[0]
  const reach = connectivity.data
  const unreachable = reach ? !reach.online || !reach.provider_reachable : false
  const noLocalModel = !local.model_name.trim()
  const missingLocalModel = isLocal && noLocalModel

  /**
   * The hosting block follows the backend rather than the mount. `defaultOpen`
   * is read once, so selecting Self-hosted used to leave the machine facts, the
   * plan, and the sbatch script folded away at the one moment they matter. The
   * disclosure is still a disclosure: the reader can fold it back.
   */
  const [hostingOpen, setHostingOpen] = useState(isLocal)
  useEffect(() => {
    setHostingOpen(isLocal)
  }, [isLocal])

  const roleSummary = (role: AgentRole) =>
    capabilities.data?.agents.find((agent) => agent.role === role)?.summary ?? ROLE_FALLBACK[role]

  /* A decision model picked for the default or for any role but the Predictor
     is held here, unsaved, until the companion step confirms or cancels it. */
  const [companion, setCompanion] = useState<CompanionRequest | null>(null)
  const decisionModels = useDecisionModels()
  const predictorIsDecision = isDecisionModelId(models.role_models.predictor, decisionModels.ids)

  return (
    <>
      <SectionHead
        title="Models and compute"
        description="Where a model call goes, which model answers it, and what runs the weights."
      />

      {unreachable && reach && (
        <Callout tone="caution" icon={<PlugZap size={15} />}>
          {reach.online
            ? 'The provider could not be reached, so the model catalog and prices may be a cached copy.'
            : 'This machine appears to be offline. The model catalog, prices, and Hugging Face search are unavailable until it is back.'}
          {reach.reason ? ` ${reach.reason}` : ''}
        </Callout>
      )}

      <Group title="Where inference runs">
        <Segmented
          value={backend}
          options={BACKENDS.map((option) => ({ value: option.value, label: option.label }))}
          onChange={(next) => update('connection', { backend: next })}
        />
        <span className="t-small secondary">{active.blurb}</span>
        {/* The service refuses to store a self-hosted backend with no weights
            behind it, so saying so here beats a rejected save and a toast. The
            checkpoint is chosen in the hosting block, which renders in both
            modes precisely so this is reachable before the switch. */}
        {!isLocal && noLocalModel && (
          <div className="row gap-2 wrap">
            <span className="t-tiny muted grow">
              Self-hosted cannot be selected yet: the service will not store a backend it has no checkpoint for.
            </span>
            <Button size="sm" variant="secondary" onClick={() => setHostingOpen(true)}>
              Choose a checkpoint
            </Button>
          </div>
        )}
      </Group>

      {missingLocalModel && (
        <Callout
          tone="critical"
          icon={<TriangleAlert size={15} />}
          title="Self-hosted inference has no model yet"
          action={
            <Button size="sm" onClick={() => update('connection', { backend: 'openrouter' })}>
              Use OpenRouter
            </Button>
          }
        >
          Every run will fail at the first model call until a repository id or a local checkout path is set. Search for
          one below, or go back to OpenRouter.
        </Callout>
      )}

      {isLocal ? (
        <>
          <Group title="Hugging Face access">
            <HuggingFaceKeyPanel />
          </Group>
          {/* Demoted rather than removed: a key still has to be addable,
              replaceable, and verifiable without switching backends first. */}
          <Disclosure
            title="OpenRouter API key"
            subtitle="Stored, but not used while the backend is self-hosted"
          >
            <OpenRouterKeyPanel links={links} heading={null} inUse={false} />
          </Disclosure>
        </>
      ) : (
        <OpenRouterKeyPanel links={links} />
      )}

      <Group title="Which model">
        {isLocal ? (
          <HfModelSearch />
        ) : (
          <>
            <ModelPicker
              value={models.default_model}
              onChange={(next) => update('models', { default_model: next })}
              onDecisionPick={(decisionId) => setCompanion({ decisionId, origin: 'default' })}
              placeholder="Choose the model every role falls back to"
            />
            {companion?.origin === 'default' && (
              <CompanionPanel request={companion} onDone={() => setCompanion(null)} />
            )}
            {row && (
              <div className="row gap-2 wrap">
                <Badge tone="accent" mono>
                  {compactNumber(row.context_length)} token context
                </Badge>
                <Badge outline mono>
                  {usdPerMillion(row.prompt_usd_per_mtok)} in
                </Badge>
                <Badge outline mono>
                  {usdPerMillion(row.completion_usd_per_mtok)} out
                </Badge>
                {row.supports_structured_output && <Badge tone="info">structured output</Badge>}
                {row.supports_tools && <Badge>tool calling</Badge>}
              </div>
            )}
            {links.openrouter_models && (
              <a className="t-tiny row gap-1" href={links.openrouter_models} target="_blank" rel="noreferrer">
                <ExternalLink size={12} /> Compare models and prices at the provider
              </a>
            )}
          </>
        )}

        {/* Nothing is billed per token on your own hardware, and the section
            has already said so, so the hint cannot talk about the rate card. */}
        <Field
          label="Reasoning effort"
          hint={
            isLocal
              ? 'Thinking tokens share the output ceiling with the answer, so a higher effort leaves less room for the reply and holds the GPU for longer.'
              : 'Thinking tokens are billed at the completion rate and share the output ceiling with the answer.'
          }
          info={
            <p>
              Off is markedly faster{isLocal ? '' : ' and cheaper'}, and is the right default here because the plan and
              the critic already supply the deliberation. Raise it for a genuinely hard prediction, or when the critic
              keeps asking for another iteration.
            </p>
          }
        >
          <Segmented
            value={models.reasoning_effort}
            options={EFFORTS}
            onChange={(next) => update('models', { reasoning_effort: next })}
          />
        </Field>

        {/* Two different things share one configuration key. A self-hosted run
            hands this value to sentence-transformers on the host; a hosted run
            sends it to the provider as an embeddings route. A provider id would
            crash the first, and a sentence-transformers id means nothing to the
            second, so the control has to follow the backend. */}
        {isLocal ? (
          <HfEmbeddingSearch />
        ) : (
          <Field
            label="Embedding model"
            hint="Deduplicates and clusters evidence before the report is written. It resolves through the hosted catalog and is billed per token, like every other call to the provider."
          >
            <ModelPicker
              embedding
              value={models.embedding_model}
              onChange={(next) => update('models', { embedding_model: next })}
              placeholder="Choose an embedding model"
            />
          </Field>
        )}

        {isLocal ? (
          <NumberSetting
            label="Context window"
            section="local"
            field="max_model_len"
            min={0}
            max={4_000_000}
            step={1024}
            width={180}
            hint={
              local.max_model_len > 0
                ? `The engine is told the checkpoint holds ${tokens(local.max_model_len)} tokens.`
                : 'Zero uses the context length the checkpoint declares.'
            }
          />
        ) : (
          <Field
            label="Context window"
            hint={
              models.context_window > 0
                ? `${tokens(models.context_window)} tokens for any model that neither the catalog nor the built-in table knows. A known model keeps its own window; force one per role below.`
                : 'Each role resolves its window from the catalog, then the built-in table. Set this for a model neither knows (zero falls back to 128K tokens); force a window per role below.'
            }
          >
            <NumberControl
              ariaLabel="Context window override"
              value={models.context_window}
              min={0}
              max={4_000_000}
              step={1024}
              width={180}
              onCommit={(next) => update('models', { context_window: next })}
            />
          </Field>
        )}
      </Group>

      {predictorIsDecision && <DecisionModelSettings />}

      <Disclosure
        title="Per role"
        subtitle="Override the model, output ceiling, and temperature for one agent"
        defaultOpen={config.appearance.show_advanced_by_default || predictorIsDecision}
      >
        <table className="settings__roles">
          <thead>
            <tr>
              <th style={{ width: '26%' }}>Role</th>
              <th>Model</th>
              <th style={{ width: 108 }}>Max output</th>
              <th style={{ width: 208 }}>Temperature</th>
              <th style={{ width: 34 }} aria-label="Reset" />
            </tr>
          </thead>
          <tbody>
            {AGENT_ROLES.map((role) => (
              <Fragment key={role}>
                <tr>
                  <td>
                    <div className="stack" style={{ gap: 1 }}>
                      <span className="t-small semibold" style={{ textTransform: 'capitalize' }}>
                        {role}
                      </span>
                      <span className="t-micro muted">{roleSummary(role)}</span>
                      {role === 'predictor' && (
                        <span className="t-micro muted">Also accepts a structured decision model.</span>
                      )}
                    </div>
                  </td>
                  <td>
                    <ModelPicker
                      compact
                      allowInherit
                      value={models.role_models[role]}
                      inheritLabel="Inherit default"
                      acceptsDecision={role === 'predictor'}
                      onDecisionPick={
                        role === 'predictor' ? undefined : (decisionId) => setCompanion({ decisionId, origin: role })
                      }
                      onChange={(next) => update('models', { role_models: { ...models.role_models, [role]: next } })}
                    />
                  </td>
                  <td>
                    <NumberControl
                      ariaLabel={`${role} max output tokens`}
                      value={models.role_max_tokens[role]}
                      min={0}
                      max={1_000_000}
                      step={500}
                      onCommit={(next) =>
                        update('models', { role_max_tokens: { ...models.role_max_tokens, [role]: next } })
                      }
                    />
                  </td>
                  <td>
                    <SliderControl
                      value={models.role_temperatures[role]}
                      min={0}
                      max={1}
                      step={0.05}
                      onCommit={(next) =>
                        update('models', { role_temperatures: { ...models.role_temperatures, [role]: next } })
                      }
                    />
                  </td>
                  <td>
                    <Tooltip content="Reset this row to the shipped default">
                      <Button
                        variant="ghost"
                        size="sm"
                        iconOnly
                        icon={<Undo2 size={13} />}
                        aria-label={`Reset ${role} to default`}
                        onClick={() =>
                          update('models', {
                            role_models: { ...models.role_models, [role]: ROLE_DEFAULTS.model[role] },
                            role_max_tokens: { ...models.role_max_tokens, [role]: ROLE_DEFAULTS.maxTokens[role] },
                            role_context_windows: {
                              ...(models.role_context_windows ?? ROLE_DEFAULTS.contextWindow),
                              [role]: 0,
                            },
                            role_temperatures: {
                              ...models.role_temperatures,
                              [role]: ROLE_DEFAULTS.temperature[role],
                            },
                          })
                        }
                      />
                    </Tooltip>
                  </td>
                </tr>
                {companion?.origin === role && (
                  <tr>
                    <td colSpan={5}>
                      <CompanionPanel request={companion} onDone={() => setCompanion(null)} />
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
        <div className="stack gap-2" style={{ marginTop: 'var(--s-5)' }}>
          <span className="eyebrow">Context window per role</span>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))',
              gap: 'var(--s-3)',
            }}
          >
            {AGENT_ROLES.map((role) => {
              const resolved = resolvedWindows[role]
              return (
                <div key={role} className="stack" style={{ gap: 2 }}>
                  <span className="t-small semibold" style={{ textTransform: 'capitalize' }}>
                    {role}
                  </span>
                  <span className="t-micro muted tabular">
                    {resolved
                      ? `${tokens(resolved.tokens)} from the ${WINDOW_SOURCES[resolved.source] ?? resolved.source}`
                      : 'Resolved when the settings are saved'}
                  </span>
                  <NumberControl
                    ariaLabel={`${role} context window override`}
                    value={models.role_context_windows?.[role] ?? 0}
                    min={0}
                    max={10_000_000}
                    step={1024}
                    width={150}
                    onCommit={(next) =>
                      update('models', {
                        role_context_windows: {
                          ...(models.role_context_windows ?? ROLE_DEFAULTS.contextWindow),
                          // 0 resolves the window; a forced window is at least 1024 tokens (the API's floor)
                          [role]: next > 0 && next < 1024 ? 1024 : next,
                        },
                      })
                    }
                  />
                </div>
              )
            })}
          </div>
        </div>
        <p className="t-tiny muted" style={{ marginTop: 'var(--s-3)' }}>
          A max output of 0 derives the ceiling from the context window, which produces a very large number on a
          long-context model. The shipped values are sized to what each role actually writes. A context window of 0
          resolves it (catalog, built-in table, then the window above); any other value forces it for that role, for
          example the max model length of a self-hosted server. A structured decision model keeps its state limit.
        </p>
      </Disclosure>

      <Disclosure
        title="Self-hosted runtime"
        subtitle={
          isLocal
            ? 'What this machine has, the launch command, and how the weights are loaded'
            : 'Choose the checkpoint and settle the hardware before switching the backend'
        }
        open={hostingOpen}
        onOpenChange={setHostingOpen}
      >
        <div className="stack gap-5">
          {/* On OpenRouter the checkpoint chooser lives here, because it is the
              only control that can set a local model name and the service will
              not accept the self-hosted backend without one. While self-hosted
              it sits under Which model instead, so it never renders twice. */}
          {!isLocal && (
            <Group title="Which checkpoint">
              <HfModelSearch />
            </Group>
          )}
          <Group title="This machine">
            <MachineFacts />
          </Group>
          <Group title="Deployment plan">
            <DeploymentPlan local={local} />
          </Group>
          <SelfHostedRuntime />
        </div>
      </Disclosure>

      <Disclosure
        title="Requests"
        subtitle={
          isLocal
            ? 'How long a model call may take, and how often it is retried'
            : 'Endpoint, request identity, and retry behaviour'
        }
        defaultOpen={config.appearance.show_advanced_by_default}
      >
        <div className="stack gap-4">
          {!isLocal && (
            <>
              <TextSetting
                label="Base URL"
                section="connection"
                field="openrouter_base_url"
                mono
                placeholder="https://openrouter.ai/api/v1"
                hint="Change this only to reach a proxy or a compatible gateway."
              />
              <Grid>
                <TextSetting
                  label="Site URL"
                  section="connection"
                  field="openrouter_site_url"
                  placeholder="https://your-lab.example"
                  hint="Sent as the HTTP-Referer header."
                />
                <TextSetting
                  label="App name"
                  section="connection"
                  field="openrouter_app_name"
                  placeholder="COMPASS"
                  hint="Sent as the X-Title header and shown in provider activity."
                />
              </Grid>
            </>
          )}
          <Grid columns={3}>
            <NumberSetting
              label="Request timeout"
              section="connection"
              field="request_timeout_seconds"
              min={10}
              max={1800}
              suffix="s"
              hint="A call is abandoned after this long."
            />
            <NumberSetting
              label="Max retries"
              section="connection"
              field="max_retries"
              min={0}
              max={10}
              hint="Retries after a transport or rate-limit failure."
            />
            <NumberSetting
              label="Retry delay"
              section="connection"
              field="retry_delay_seconds"
              min={0}
              max={60}
              step={0.5}
              integer={false}
              suffix="s"
              hint="Waited before the first retry, then backed off."
            />
          </Grid>
        </div>
      </Disclosure>
    </>
  )
}
