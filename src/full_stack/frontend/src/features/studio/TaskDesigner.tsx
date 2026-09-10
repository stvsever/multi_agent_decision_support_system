/**
 * Prediction task designer.
 *
 * The engine accepts five task families. Four are flat and need only labels;
 * the fifth is a tree of mixed-mode nodes. One form serves all five: the family
 * decides which controls appear and what "runnable" means, and every problem is
 * shown against the control that caused it as soon as that control is touched.
 */

import clsx from 'clsx'
import { Check, Network, Plus, Trash2 } from 'lucide-react'
import { useId, useState, type AriaAttributes, type ReactNode } from 'react'
import { Badge, Button, EmptyState, Field, Input } from '@/components/ui/primitives'
import type { PredictionType, TaskNodeInput, TaskSpecInput } from '@/lib/types'
import { ChipList } from './ChipList'
import { TaskTree, newNode } from './TaskTree'
import { TaskTreeSource } from './TaskTreeSource'
import {
  FAMILY_LABELS,
  countNodes,
  problemsFor,
  slugify,
  walkNodes,
  type TaskProblem,
  type TaskReport,
} from './taskValidation'

type FlatFieldName = 'target_label' | 'control_label' | 'class_labels' | 'regression_outputs'

/** What TaskField hands its control so the message reaches assistive tech. */
type FieldAria = Pick<AriaAttributes, 'aria-invalid' | 'aria-describedby'>

export function TaskDesigner({
  task,
  report,
  onChange,
  types,
  onNotify,
}: {
  task: TaskSpecInput
  report: TaskReport
  onChange: (patch: Partial<TaskSpecInput>) => void
  types: { value: PredictionType; label: string; summary: string; needs: string[] }[]
  onNotify?: (title: string, body: string) => void
}) {
  /* A pristine field states its requirement as a hint. It only turns into an
     error once it has been touched or already carries something. */
  const [touched, setTouched] = useState<Record<string, boolean>>({})
  const touch = (field: FlatFieldName) => setTouched((state) => (state[field] ? state : { ...state, [field]: true }))

  const hasContent = (field: FlatFieldName): boolean => {
    switch (field) {
      case 'target_label':
        return task.target_label.trim().length > 0
      case 'control_label':
        return task.control_label.trim().length > 0
      case 'class_labels':
        return task.class_labels.length > 0
      case 'regression_outputs':
        return task.regression_outputs.length > 0
    }
  }

  const errorFor = (field: FlatFieldName): TaskProblem | undefined => {
    const problem = problemsFor(report.problems, field).find((entry) => entry.severity === 'error')
    if (!problem) return undefined
    return touched[field] || hasContent(field) ? problem : undefined
  }

  /* A warning is about a value the form does not show, so it cannot wait for
     the field to be touched. */
  const warningFor = (field: FlatFieldName): TaskProblem | undefined =>
    problemsFor(report.problems, field).find((entry) => entry.severity === 'warning')

  const chooseFamily = (value: PredictionType) => {
    const patch: Partial<TaskSpecInput> = { prediction_type: value }
    if (value === 'hierarchical' && !task.root) {
      patch.root = newNode({
        node_id: slugify(task.target_label) || 'root',
        display_name: task.target_label.trim() || 'Phenotype profile',
      })
    }
    // Binary reads two stored class labels in preference to the target and the
    // comparator, so leaving a multiclass draft behind would quietly win.
    if (value === 'binary') patch.class_labels = []
    onChange(patch)
  }

  return (
    <div className="stack gap-5">
      <Field
        label="Task family"
        info={
          <>
            <p>
              The family fixes the contract the Predictor must satisfy and the checklist the Critic scores it
              against.
            </p>
            <ol>
              <li>Classification returns a label and a probability per class.</li>
              <li>Regression returns one or more numeric values.</li>
              <li>
                A tree mixes families: a root question with dependent sub-questions, each answered in its own mode.
              </li>
            </ol>
          </>
        }
      >
        <div className="typegrid">
          {types.map((type) => (
            <button
              key={type.value}
              type="button"
              className="typecard"
              data-selected={task.prediction_type === type.value}
              onClick={() => chooseFamily(type.value)}
            >
              <span className="typecard__label">{type.label}</span>
              <span className="typecard__summary">{type.summary}</span>
            </button>
          ))}
        </div>
      </Field>

      <RequirementStrip report={report} family={task.prediction_type} />

      {task.prediction_type !== 'hierarchical' ? (
        <div className="grid grid--2">
          <TaskField
            label="Target label"
            hint="The phenotype being predicted. This name reaches every agent prompt."
            error={errorFor('target_label')}
          >
            {(aria) => (
              <Input
                {...aria}
                value={task.target_label}
                invalid={Boolean(errorFor('target_label'))}
                onChange={(event) => onChange({ target_label: event.target.value })}
                onBlur={() => touch('target_label')}
                placeholder="for example: major_depressive_episode"
              />
            )}
          </TaskField>

          {task.prediction_type === 'binary' && (
            <TaskField
              label="Comparator label"
              hint="What the target is being distinguished from."
              error={errorFor('control_label')}
              warning={warningFor('control_label')}
            >
              {(aria) => (
                <Input
                  {...aria}
                  value={task.control_label}
                  invalid={Boolean(errorFor('control_label'))}
                  onChange={(event) => onChange({ control_label: event.target.value })}
                  onBlur={() => touch('control_label')}
                  placeholder="for example: healthy_comparator"
                />
              )}
            </TaskField>
          )}

          {task.prediction_type === 'multiclass' && (
            <TaskField
              label="Class labels"
              hint="Three or more mutually exclusive classes."
              error={errorFor('class_labels')}
            >
              {(aria) => (
                <ChipList
                  {...aria}
                  aria-label="Class labels"
                  values={task.class_labels}
                  invalid={Boolean(errorFor('class_labels'))}
                  onChange={(class_labels) => onChange({ class_labels })}
                  onBlurCapture={() => touch('class_labels')}
                  placeholder="Add a class, then Enter"
                />
              )}
            </TaskField>
          )}

          {task.prediction_type.startsWith('regression') && (
            <TaskField
              label={task.prediction_type === 'regression_univariate' ? 'Output name' : 'Output names'}
              hint={
                task.prediction_type === 'regression_univariate'
                  ? 'Exactly one continuous output.'
                  : 'Two or more continuous outputs, predicted together.'
              }
              error={errorFor('regression_outputs')}
            >
              {(aria) => (
                <ChipList
                  {...aria}
                  aria-label={task.prediction_type === 'regression_univariate' ? 'Output name' : 'Output names'}
                  values={task.regression_outputs}
                  invalid={Boolean(errorFor('regression_outputs'))}
                  onChange={(regression_outputs) => onChange({ regression_outputs })}
                  onBlurCapture={() => touch('regression_outputs')}
                  placeholder="Add an output, then Enter"
                />
              )}
            </TaskField>
          )}
        </div>
      ) : (
        <TreeSection task={task} report={report} onChange={onChange} onNotify={onNotify} />
      )}
    </div>
  )
}

/* --- Pieces ---------------------------------------------------------------- */

/**
 * A labelled control whose message is wired to it rather than merely placed
 * next to it. Colour alone does not reach a screen reader, so the control
 * carries aria-invalid and points at whichever message is on screen: the
 * hint while the field is clean, the error once it is not.
 */
function TaskField({
  label,
  hint,
  error,
  warning,
  children,
}: {
  label: string
  hint: string
  error?: TaskProblem
  warning?: TaskProblem
  children: (aria: FieldAria) => ReactNode
}) {
  const hintId = useId()
  const errorId = useId()
  const warningId = useId()
  const describedBy = [error ? errorId : hintId, warning ? warningId : null].filter(Boolean).join(' ')

  return (
    <Field
      label={label}
      required
      hint={<span id={hintId}>{hint}</span>}
      error={error ? <span id={errorId}>{error.message}</span> : undefined}
    >
      <div className="stack gap-1">
        {children({ 'aria-invalid': error ? true : undefined, 'aria-describedby': describedBy })}
        {warning && (
          <span id={warningId} className="t-tiny studio__warn">
            {warning.message}
          </span>
        )}
      </div>
    </Field>
  )
}

/** What this family still needs, as a row of met and unmet requirements. */
function RequirementStrip({ report, family }: { report: TaskReport; family: PredictionType }) {
  return (
    <div className="taskreq">
      <span className="t-tiny muted" style={{ flex: 'none' }}>
        {FAMILY_LABELS[family]} needs
      </span>
      {report.requirements.map((requirement) => (
        <span key={requirement.field} className={clsx('taskreq__pill', requirement.met && 'taskreq__pill--met')}>
          {requirement.met ? <Check size={11} /> : <span className="taskreq__dot" />}
          <span>{requirement.label}</span>
          {requirement.note && <span className="faint tabular">{requirement.note}</span>}
        </span>
      ))}
    </div>
  )
}

function TreeSection({
  task,
  report,
  onChange,
  onNotify,
}: {
  task: TaskSpecInput
  report: TaskReport
  onChange: (patch: Partial<TaskSpecInput>) => void
  onNotify?: (title: string, body: string) => void
}) {
  let depth = 0
  walkNodes(task.root, (_node, _key, level) => {
    depth = Math.max(depth, level)
  })

  const apply = (root: TaskNodeInput, origin: string) => {
    onChange({ root })
    onNotify?.('Task tree applied', `${countNodes(root)} nodes read from ${origin}.`)
  }

  return (
    <div className="stack gap-4">
      <div className="row between gap-3 wrap">
        <span className="stack" style={{ gap: 1 }}>
          <span className="t-small semibold">Task tree</span>
          <span className="t-tiny muted">
            Each node is answered in its own mode, in the context of its parent.
          </span>
        </span>
        <span className="row gap-2">
          {task.root && (
            <>
              <Badge mono>
                {report.nodeCount} node{report.nodeCount === 1 ? '' : 's'}
              </Badge>
              <Badge mono>depth {depth}</Badge>
              <Button
                size="sm"
                variant="ghost"
                icon={<Trash2 size={13} />}
                onClick={() => onChange({ root: null })}
              >
                Clear
              </Button>
            </>
          )}
        </span>
      </div>

      <TaskTreeSource root={task.root ?? null} onApply={apply} />

      {task.root ? (
        <TaskTree root={task.root} problems={report.problems} onChange={(root) => onChange({ root })} />
      ) : (
        <EmptyState
          icon={<Network size={20} />}
          title="No tree yet"
          body="Drop a file, apply an example, or start from a single root question and add children to it."
          action={
            <Button
              variant="primary"
              icon={<Plus size={14} />}
              onClick={() =>
                onChange({
                  root: newNode({
                    node_id: slugify(task.target_label) || 'root',
                    display_name: task.target_label.trim() || 'Phenotype profile',
                  }),
                })
              }
            >
              Add the root question
            </Button>
          }
        />
      )}
    </div>
  )
}
