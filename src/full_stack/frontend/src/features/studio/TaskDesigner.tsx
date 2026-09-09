/**
 * Prediction task designer.
 *
 * The engine accepts five task families. Four are flat and need only labels;
 * the fifth is a tree of mixed-mode nodes. Rather than five separate forms this
 * is one form whose required fields follow the chosen family, so the shape of
 * the question stays visible while the detail changes.
 */

import { Plus, Trash2, X } from 'lucide-react'
import { useCallback } from 'react'
import { Badge, Button, Callout, Field, Input, Select } from '@/components/ui/primitives'
import type { NodeMode, PredictionType, TaskNodeInput, TaskSpecInput } from '@/lib/types'

const MODE_LABELS: Record<NodeMode, string> = {
  binary_classification: 'Binary',
  multiclass_classification: 'Multiclass',
  univariate_regression: 'Univariate regression',
  multivariate_regression: 'Multivariate regression',
}

/** The engine validates these contracts, so mirror them here for fast feedback. */
export function validateTask(task: TaskSpecInput): string[] {
  const problems: string[] = []
  const label = task.target_label.trim()
  if (!label) problems.push('Give the target a name.')

  if (task.prediction_type === 'binary' && !task.control_label.trim()) {
    problems.push('Binary classification needs a comparator label.')
  }
  if (task.prediction_type === 'multiclass' && task.class_labels.filter(Boolean).length < 3) {
    problems.push('Multiclass needs at least three class labels.')
  }
  if (task.prediction_type === 'regression_univariate' && task.regression_outputs.filter(Boolean).length !== 1) {
    problems.push('Univariate regression needs exactly one output name.')
  }
  if (task.prediction_type === 'regression_multivariate' && task.regression_outputs.filter(Boolean).length < 2) {
    problems.push('Multivariate regression needs at least two output names.')
  }
  if (task.prediction_type === 'hierarchical') {
    if (!task.root) problems.push('Add a root node to the task tree.')
    else problems.push(...validateNode(task.root, new Set()))
  }
  return problems
}

function validateNode(node: TaskNodeInput, seen: Set<string>): string[] {
  const problems: string[] = []
  const id = node.node_id.trim()
  if (!id) problems.push('Every node needs an id.')
  else if (seen.has(id)) problems.push(`Duplicate node id: ${id}`)
  else seen.add(id)
  if (!node.display_name.trim()) problems.push(`Node "${id || 'unnamed'}" needs a display name.`)

  const classes = node.class_labels.filter(Boolean)
  const outputs = node.regression_outputs.filter(Boolean)
  if (node.mode === 'binary_classification' && classes.length !== 2) {
    problems.push(`"${id}" is binary and needs exactly two class labels.`)
  }
  if (node.mode === 'multiclass_classification' && classes.length < 3) {
    problems.push(`"${id}" is multiclass and needs at least three class labels.`)
  }
  if (node.mode === 'univariate_regression' && outputs.length !== 1) {
    problems.push(`"${id}" is univariate and needs exactly one output.`)
  }
  if (node.mode === 'multivariate_regression' && outputs.length < 2) {
    problems.push(`"${id}" is multivariate and needs at least two outputs.`)
  }
  for (const child of node.children) problems.push(...validateNode(child, seen))
  return problems
}

export function newNode(seed: Partial<TaskNodeInput> = {}): TaskNodeInput {
  return {
    node_id: seed.node_id ?? `node_${Math.random().toString(36).slice(2, 7)}`,
    display_name: seed.display_name ?? 'New node',
    mode: seed.mode ?? 'binary_classification',
    class_labels: seed.class_labels ?? ['CASE', 'CONTROL'],
    regression_outputs: seed.regression_outputs ?? [],
    unit_by_output: seed.unit_by_output ?? {},
    required: seed.required ?? true,
    children: seed.children ?? [],
  }
}

/* --- Chip list ------------------------------------------------------------ */

function ChipList({
  values,
  onChange,
  placeholder,
  min,
}: {
  values: string[]
  onChange: (next: string[]) => void
  placeholder: string
  min?: number
}) {
  const add = useCallback(
    (raw: string) => {
      const parts = raw
        .split(',')
        .map((p) => p.trim())
        .filter(Boolean)
      if (!parts.length) return
      const next = [...values]
      for (const part of parts) if (!next.includes(part)) next.push(part)
      onChange(next)
    },
    [values, onChange],
  )

  return (
    <div className="chiplist">
      {values.map((value, index) => (
        <span key={`${value}-${index}`} className="chiplist__chip">
          <span className="mono truncate">{value}</span>
          <button
            type="button"
            aria-label={`Remove ${value}`}
            onClick={() => onChange(values.filter((_, i) => i !== index))}
            disabled={min !== undefined && values.length <= min}
          >
            <X size={11} />
          </button>
        </span>
      ))}
      <input
        className="chiplist__input"
        placeholder={placeholder}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ',') {
            event.preventDefault()
            add(event.currentTarget.value)
            event.currentTarget.value = ''
          }
          if (event.key === 'Backspace' && !event.currentTarget.value && values.length) {
            onChange(values.slice(0, -1))
          }
        }}
        onBlur={(event) => {
          add(event.currentTarget.value)
          event.currentTarget.value = ''
        }}
      />
    </div>
  )
}

/* --- Node editor ---------------------------------------------------------- */

function NodeEditor({
  node,
  depth,
  onChange,
  onRemove,
}: {
  node: TaskNodeInput
  depth: number
  onChange: (next: TaskNodeInput) => void
  onRemove?: () => void
}) {
  const patch = (delta: Partial<TaskNodeInput>) => onChange({ ...node, ...delta })
  const needsClasses = node.mode.endsWith('classification')

  return (
    <div className="tasknode" style={{ marginLeft: depth * 18 }}>
      <div className="tasknode__head">
        <Badge tone={depth === 0 ? 'accent' : 'neutral'} mono>
          {depth === 0 ? 'root' : `depth ${depth}`}
        </Badge>
        <Input
          value={node.display_name}
          onChange={(e) => patch({ display_name: e.target.value })}
          placeholder="Display name"
          style={{ maxWidth: 210 }}
        />
        <Input
          mono
          value={node.node_id}
          onChange={(e) => patch({ node_id: e.target.value.replace(/\s+/g, '_') })}
          placeholder="node_id"
          style={{ maxWidth: 150 }}
        />
        <Select
          value={node.mode}
          onChange={(e) => {
            const mode = e.target.value as NodeMode
            patch({
              mode,
              class_labels: mode === 'binary_classification' ? ['CASE', 'CONTROL'] : mode === 'multiclass_classification' ? node.class_labels : [],
              regression_outputs: mode.endsWith('regression') ? node.regression_outputs : [],
            })
          }}
          style={{ maxWidth: 190 }}
        >
          {Object.entries(MODE_LABELS).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </Select>
        <div className="grow" />
        <Button
          size="sm"
          variant="ghost"
          icon={<Plus size={13} />}
          onClick={() => patch({ children: [...node.children, newNode({ display_name: 'Child node' })] })}
        >
          Child
        </Button>
        {onRemove && (
          <Button size="sm" variant="ghost" iconOnly icon={<Trash2 size={13} />} onClick={onRemove} aria-label="Remove node" />
        )}
      </div>

      <div className="tasknode__body">
        {needsClasses ? (
          <Field label={node.mode === 'binary_classification' ? 'Two class labels' : 'Three or more class labels'}>
            <ChipList
              values={node.class_labels}
              onChange={(class_labels) => patch({ class_labels })}
              placeholder="Add a label, then Enter"
            />
          </Field>
        ) : (
          <Field
            label={node.mode === 'univariate_regression' ? 'One output name' : 'Two or more output names'}
            hint="Optionally append a unit in the unit column below."
          >
            <ChipList
              values={node.regression_outputs}
              onChange={(regression_outputs) => patch({ regression_outputs })}
              placeholder="Add an output, then Enter"
            />
          </Field>
        )}
      </div>

      {node.children.map((child, index) => (
        <NodeEditor
          key={`${child.node_id}-${index}`}
          node={child}
          depth={depth + 1}
          onChange={(next) => patch({ children: node.children.map((c, i) => (i === index ? next : c)) })}
          onRemove={() => patch({ children: node.children.filter((_, i) => i !== index) })}
        />
      ))}
    </div>
  )
}

/* --- Designer ------------------------------------------------------------- */

export function TaskDesigner({
  task,
  onChange,
  types,
}: {
  task: TaskSpecInput
  onChange: (patch: Partial<TaskSpecInput>) => void
  types: { value: PredictionType; label: string; summary: string; needs: string[] }[]
}) {
  const problems = validateTask(task)

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
                A hierarchical tree mixes families: a root question with dependent sub-questions, each answered in
                its own mode.
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
              onClick={() => {
                const patch: Partial<TaskSpecInput> = { prediction_type: type.value }
                if (type.value === 'hierarchical' && !task.root) {
                  patch.root = newNode({
                    node_id: 'root',
                    display_name: task.target_label || 'Phenotype profile',
                    mode: 'binary_classification',
                  })
                }
                onChange(patch)
              }}
            >
              <span className="typecard__label">{type.label}</span>
              <span className="typecard__summary">{type.summary}</span>
            </button>
          ))}
        </div>
      </Field>

      {task.prediction_type !== 'hierarchical' ? (
        <div className="grid grid--2">
          <Field
            label="Target label"
            required
            hint="The phenotype being predicted. This name reaches every agent prompt."
          >
            <Input
              value={task.target_label}
              onChange={(e) => onChange({ target_label: e.target.value })}
              placeholder="for example: major_depressive_episode"
            />
          </Field>

          {task.prediction_type === 'binary' && (
            <Field label="Comparator label" required hint="What the target is being distinguished from.">
              <Input
                value={task.control_label}
                onChange={(e) => onChange({ control_label: e.target.value })}
                placeholder="for example: healthy_comparator"
              />
            </Field>
          )}

          {task.prediction_type === 'multiclass' && (
            <Field label="Class labels" required hint="Three or more mutually exclusive classes.">
              <ChipList
                values={task.class_labels}
                onChange={(class_labels) => onChange({ class_labels })}
                placeholder="Add a class, then Enter"
              />
            </Field>
          )}

          {task.prediction_type.startsWith('regression') && (
            <Field
              label={task.prediction_type === 'regression_univariate' ? 'Output name' : 'Output names'}
              required
              hint={
                task.prediction_type === 'regression_univariate'
                  ? 'Exactly one continuous output.'
                  : 'Two or more continuous outputs, predicted together.'
              }
            >
              <ChipList
                values={task.regression_outputs}
                onChange={(regression_outputs) => onChange({ regression_outputs })}
                placeholder="Add an output, then Enter"
              />
            </Field>
          )}
        </div>
      ) : (
        <div className="stack gap-3">
          <Field label="Task tree" hint="Each node is answered in its own mode. Children are answered in context of their parent.">
            <div />
          </Field>
          {task.root ? (
            <NodeEditor node={task.root} depth={0} onChange={(root) => onChange({ root })} />
          ) : (
            <Button
              icon={<Plus size={14} />}
              onClick={() => onChange({ root: newNode({ node_id: 'root', display_name: 'Phenotype profile' }) })}
            >
              Add the root node
            </Button>
          )}
        </div>
      )}

      {problems.length > 0 && (
        <Callout tone="caution" title="This task is not runnable yet">
          <ul className="stack gap-1" style={{ marginTop: 4 }}>
            {problems.map((problem) => (
              <li key={problem}>{problem}</li>
            ))}
          </ul>
        </Callout>
      )}
    </div>
  )
}
