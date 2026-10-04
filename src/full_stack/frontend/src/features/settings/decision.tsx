/**
 * Structured decision models in settings.
 *
 * A decision model answers typed questions and writes no text, so it can only
 * serve the Predictor. Choosing one anywhere else does not save it there: it
 * opens the companion step, which is the settings equivalent of the command
 * line prompt that asks for a conventional LLM to run the other agents.
 */

import { Target, Undo2 } from 'lucide-react'
import { useState } from 'react'
import { ModelPicker } from '@/components/ui/ModelPicker'
import { Badge, Button, Callout, Field, Segmented, Tooltip } from '@/components/ui/primitives'
import {
  COMPANION_FALLBACK_MODEL,
  DECISION_DEFAULTS,
  DECISION_TOOLTIP,
  decisionLabel,
  isDecisionModelId,
  useDecisionModels,
} from '@/lib/decision'
import type { DecisionProvider } from '@/lib/types'
import { Grid, Group, SliderControl, SwitchRow } from './controls'
import { useSettingsController } from './state'

const PROVIDERS: { value: DecisionProvider; label: string }[] = [
  { value: 'openrouter', label: 'OpenRouter' },
  { value: 'typesafe', label: 'TypeSafe' },
]

/** A decision model somebody tried to put where it cannot serve. */
export interface CompanionRequest {
  decisionId: string
  /** `default` for the default model, otherwise the role it was picked for. */
  origin: string
}

/**
 * The inline confirm step. On confirm the decision model becomes the Predictor
 * and the chosen LLM becomes the default every other role falls back to, in one
 * save, so the service never sees a configuration it would reject.
 */
export function CompanionPanel({ request, onDone }: { request: CompanionRequest; onDone: () => void }) {
  const { config, update } = useSettingsController()
  const { specs, ids } = useDecisionModels()
  const models = config.models
  const label = decisionLabel(request.decisionId, specs)

  const current = models.default_model
  const [companion, setCompanion] = useState(() =>
    current && !isDecisionModelId(current, ids) ? current : COMPANION_FALLBACK_MODEL,
  )

  const confirm = () => {
    update('models', {
      default_model: companion,
      role_models: { ...models.role_models, predictor: request.decisionId },
    })
    onDone()
  }

  return (
    <Callout
      tone="info"
      icon={<Target size={15} />}
      title={
        <span className="row gap-2 wrap">
          <span>Use {label} as the Predictor</span>
          <Tooltip content={DECISION_TOOLTIP}>
            <Badge tone="accent">Decision model</Badge>
          </Tooltip>
        </span>
      }
    >
      <div className="stack gap-3">
        <span>
          {label} can only serve the Predictor. Use it as Predictor and keep a conventional LLM for the other agents.
        </span>
        <Field
          label="Conventional LLM for the other agents"
          hint="The Orchestrator, the tools, the Integrator, the Critic, and the Communicator fall back to this model."
        >
          <ModelPicker
            value={companion}
            onChange={setCompanion}
            placeholder="Choose the LLM the other agents use"
          />
        </Field>
        <div className="row gap-2 wrap">
          <Button size="sm" variant="primary" disabled={!companion} onClick={confirm}>
            Use as Predictor
          </Button>
          <Button size="sm" variant="ghost" onClick={onDone}>
            Cancel
          </Button>
          <span className="t-tiny muted grow">
            {request.origin === 'default' || request.origin === 'predictor'
              ? 'Nothing is saved until you confirm.'
              : `The ${request.origin} keeps its current model.`}
          </span>
        </div>
      </div>
    </Callout>
  )
}

/**
 * How the decision model is questioned. Shown only while the Predictor role is
 * a decision model, because none of it applies to a conventional Predictor.
 */
export function DecisionModelSettings() {
  const { config, update } = useSettingsController()
  const { specs } = useDecisionModels()
  const models = config.models
  const decision = { ...DECISION_DEFAULTS, ...(config.decision ?? {}) }
  const predictor = models.role_models.predictor

  return (
    <Group title="Decision model">
      <div className="row gap-2 wrap">
        <Tooltip content={DECISION_TOOLTIP}>
          <Badge tone="accent">Decision model</Badge>
        </Tooltip>
        <span className="t-small semibold">{decisionLabel(predictor, specs)}</span>
        <span className="t-tiny muted mono grow truncate">{predictor}</span>
        <Tooltip content="Put the Predictor back on the default model">
          <Button
            size="sm"
            variant="ghost"
            icon={<Undo2 size={13} />}
            onClick={() => update('models', { role_models: { ...models.role_models, predictor: '' } })}
          >
            Use a conventional Predictor
          </Button>
        </Tooltip>
      </div>
      <span className="t-small secondary">
        It serves the Predictor only. The other agents use {models.default_model || 'the default model'}.
      </span>

      <Field
        label="Provider"
        hint={
          decision.provider === 'typesafe'
            ? 'Calls the TypeSafe API directly. The service needs TYPESAFE_API_KEY in its environment.'
            : 'Routed through OpenRouter and billed to the OpenRouter key above.'
        }
      >
        <Segmented
          value={decision.provider}
          options={PROVIDERS}
          onChange={(next) => update('decision', { provider: next })}
        />
      </Field>

      <Grid>
        <Field
          label={`Option orders per Choice: ${decision.choice_orders}`}
          hint="Each multiple-choice question is asked in this many option orders and averaged, which cancels any preference for the first option."
        >
          <SliderControl
            value={decision.choice_orders}
            min={1}
            max={6}
            onCommit={(next) => update('decision', { choice_orders: next })}
          />
        </Field>
        <Field
          label={`Score levels: ${decision.score_levels}`}
          hint="How many ordered steps a continuous output is split into on the first pass."
        >
          <SliderControl
            value={decision.score_levels}
            min={2}
            max={10}
            onCommit={(next) => update('decision', { score_levels: next })}
          />
        </Field>
      </Grid>

      <SwitchRow
        label="Zoomed refinement for continuous outputs"
        hint="Asks once more inside the most likely step, so an estimate is not limited to one coarse step. Adds one request."
        checked={decision.regression_refine}
        onChange={(next) => update('decision', { regression_refine: next })}
      />

      <Grid>
        <Field
          label="Stability threshold"
          hint="The Critic rejects a prediction whose answers move more than this between option orders."
        >
          <SliderControl
            value={decision.stability_threshold}
            min={0.05}
            max={0.5}
            step={0.01}
            format={(value) => value.toFixed(2)}
            onCommit={(next) => update('decision', { stability_threshold: next })}
          />
        </Field>
        <Field
          label="Evidence sufficiency gate"
          hint={
            decision.sufficiency_threshold > 0
              ? `The Critic rejects a prediction when the model judges the record sufficient with less than ${decision.sufficiency_threshold.toFixed(2)}.`
              : 'Zero only reports whether the model judged the record sufficient; it never rejects on it.'
          }
        >
          <SliderControl
            value={decision.sufficiency_threshold}
            min={0}
            max={0.9}
            step={0.05}
            format={(value) => value.toFixed(2)}
            onCommit={(next) => update('decision', { sufficiency_threshold: next })}
          />
        </Field>
      </Grid>

      <Field
        label="Compiler model"
        hint="Writes the question book (label definitions and output scales) once per task, which is then cached."
      >
        <ModelPicker
          allowInherit
          inheritLabel="Orchestrator model"
          inheritNote="Uses the model the Orchestrator runs on"
          value={decision.compiler_model}
          onChange={(next) => update('decision', { compiler_model: next })}
        />
      </Field>
    </Group>
  )
}
