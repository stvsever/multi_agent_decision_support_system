/** Which model each part of the engine uses, and how hard it is asked to think. */

import { ExternalLink, Undo2 } from 'lucide-react'
import { useMemo } from 'react'
import { ModelPicker } from '@/components/ui/ModelPicker'
import { Badge, Button, Disclosure, Field, Segmented, Tooltip } from '@/components/ui/primitives'
import { compactNumber, tokens, usdPerMillion } from '@/lib/format'
import { useCapabilities, useCatalog } from '@/lib/hooks'
import { AGENT_ROLES, type AgentRole, type ReasoningEffort, type RoleMap } from '@/lib/types'
import { Group, NumberControl, SectionHead, SliderControl } from '../controls'
import { useSettingsController } from '../state'

const EFFORTS: { value: ReasoningEffort; label: string }[] = [
  { value: 'provider_default', label: 'Provider default' },
  { value: 'off', label: 'Off' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
]

const ROLE_DEFAULTS: { model: RoleMap<string>; maxTokens: RoleMap<number>; temperature: RoleMap<number> } = {
  model: { orchestrator: '', integrator: '', predictor: '', critic: '', communicator: '', tool: '' },
  maxTokens: { orchestrator: 8000, integrator: 8000, predictor: 12000, critic: 6000, communicator: 16000, tool: 4000 },
  temperature: { orchestrator: 0.3, integrator: 0.3, predictor: 0.2, critic: 0.2, communicator: 0.2, tool: 0.5 },
}

const ROLE_FALLBACK: Record<AgentRole, string> = {
  orchestrator: 'Reads the coverage map and writes the plan of tool steps.',
  integrator: 'Fuses the step outputs into one evidence representation.',
  predictor: 'Produces the phenotype outputs and the evidence chain.',
  critic: 'Scores the prediction and decides whether to iterate.',
  communicator: 'Writes the deep phenotype report.',
  tool: 'Runs the individual analysis tools that make up each plan step.',
}

export function ModelsSection() {
  const { config, update } = useSettingsController()
  const models = config.models
  const catalog = useCatalog({ embedding: false, limit: 500 })
  const capabilities = useCapabilities()

  const row = useMemo(
    () => catalog.data?.models.find((entry) => entry.id === models.default_model),
    [catalog.data, models.default_model],
  )

  const catalogLink = capabilities.data?.provider_links?.openrouter_models ?? ''

  const roleSummary = (role: AgentRole) =>
    capabilities.data?.agents.find((agent) => agent.role === role)?.summary ?? ROLE_FALLBACK[role]

  const costNote = row
    ? row.is_free
      ? 'This model is free at the provider, so a run costs nothing beyond your time.'
      : `Every million prompt tokens costs ${usdPerMillion(row.prompt_usd_per_mtok)} and every million completion tokens costs ${usdPerMillion(row.completion_usd_per_mtok)}; a single participant usually moves a few hundred thousand tokens.`
    : 'Prices appear once the model catalog resolves this id.'

  return (
    <>
      <SectionHead
        title="Models"
        description="The default model, how much deliberation it does, and any per-role exceptions."
      />

      <Group title="Default model">
        <ModelPicker
          value={models.default_model}
          onChange={(next) => update('models', { default_model: next })}
          placeholder="Choose the model every role falls back to"
        />
        <div className="row gap-3 wrap">
          {row && (
            <>
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
            </>
          )}
        </div>
        <span className="t-small secondary">{costNote}</span>
        {catalogLink && (
          <a className="t-tiny row gap-1" href={catalogLink} target="_blank" rel="noreferrer">
            <ExternalLink size={12} /> Compare models and prices at the provider
          </a>
        )}
      </Group>

      <Group title="Reasoning effort">
        <Field
          info={
            <>
              <p>
                Reasoning tokens are the model thinking before it answers. They are billed at the completion rate and
                they count against the same output ceiling as the answer, so a high setting is slower, more expensive,
                and can crowd out the response itself.
              </p>
              <p>
                Off is markedly faster and cheaper and is the right default for this engine, because the plan and the
                critic already supply the deliberation. Raise it for a genuinely hard prediction, or when the critic
                keeps asking for another iteration.
              </p>
            </>
          }
          label="Effort"
        >
          <Segmented
            value={models.reasoning_effort}
            options={EFFORTS}
            onChange={(next) => update('models', { reasoning_effort: next })}
          />
        </Field>
      </Group>

      <Group title="Embeddings">
        <Field
          label="Embedding model"
          hint="Used to deduplicate and cluster evidence before the report is written."
        >
          <ModelPicker
            embedding
            value={models.embedding_model}
            onChange={(next) => update('models', { embedding_model: next })}
            placeholder="Choose an embedding model"
          />
        </Field>
      </Group>

      <Disclosure
        title="Per role"
        subtitle="Override the model, output ceiling, and temperature for one agent"
        defaultOpen={config.appearance.show_advanced_by_default}
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
              <tr key={role}>
                <td>
                  <div className="stack" style={{ gap: 1 }}>
                    <span className="t-small semibold" style={{ textTransform: 'capitalize' }}>
                      {role}
                    </span>
                    <span className="t-micro muted">{roleSummary(role)}</span>
                  </div>
                </td>
                <td>
                  <ModelPicker
                    compact
                    allowInherit
                    value={models.role_models[role]}
                    inheritLabel="Inherit default"
                    onChange={(next) =>
                      update('models', { role_models: { ...models.role_models, [role]: next } })
                    }
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
            ))}
          </tbody>
        </table>
        <p className="t-tiny muted" style={{ marginTop: 'var(--s-3)' }}>
          A max output of 0 derives the ceiling from the context window, which produces a very large number on a
          long-context model and makes plans slow for no benefit. The shipped values are sized to what each role
          actually writes.
        </p>
      </Disclosure>

      <Group title="Context window">
        <Field
          label="Override"
          hint={
            models.context_window > 0
              ? `Forced to ${tokens(models.context_window)} tokens for every model.`
              : 'Zero resolves the window from the catalog entry for whichever model a role uses.'
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
      </Group>
    </>
  )
}
