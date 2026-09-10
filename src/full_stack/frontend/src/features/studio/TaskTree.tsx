/**
 * The hierarchical task editor.
 *
 * A tree of questions is only legible if it looks like a tree, so the nodes
 * carry connector lines, collapse where they get deep, and say what is wrong
 * on the node that is wrong. Ids follow the display name until someone edits
 * one by hand, which keeps a readable id without taking the field away.
 */

import clsx from 'clsx'
import { ChevronRight, CornerDownRight, Plus, Trash2 } from 'lucide-react'
import { useId, useState } from 'react'
import { Badge, Button, Input, Select, Toggle, Tooltip } from '@/components/ui/primitives'
import type { NodeMode, TaskNodeInput } from '@/lib/types'
import { ChipList } from './ChipList'
import { MODE_LABELS, MODE_ORDER, cleanList, nodeField, problemsFor, slugify, type TaskProblem } from './taskValidation'

export function newNode(seed: Partial<TaskNodeInput> = {}): TaskNodeInput {
  const display = seed.display_name ?? 'New question'
  return {
    node_id: seed.node_id ?? (slugify(display) || 'node'),
    display_name: display,
    mode: seed.mode ?? 'binary_classification',
    class_labels: seed.class_labels ?? ['CASE', 'CONTROL'],
    regression_outputs: seed.regression_outputs ?? [],
    unit_by_output: seed.unit_by_output ?? {},
    required: seed.required ?? true,
    children: seed.children ?? [],
  }
}

/* --- Immutable edits ------------------------------------------------------- */

const pathOf = (key: string): number[] => key.split('.').slice(1).map(Number)

function atPath(node: TaskNodeInput, path: number[], apply: (target: TaskNodeInput) => TaskNodeInput): TaskNodeInput {
  if (path.length === 0) return apply(node)
  const [head, ...rest] = path
  return {
    ...node,
    children: node.children.map((child, index) => (index === head ? atPath(child, rest, apply) : child)),
  }
}

function dropAt(node: TaskNodeInput, path: number[]): TaskNodeInput {
  const index = path[path.length - 1]
  return atPath(node, path.slice(0, -1), (parent) => ({
    ...parent,
    children: parent.children.filter((_, position) => position !== index),
  }))
}

/* --- Tree ------------------------------------------------------------------ */

export function TaskTree({
  root,
  problems,
  onChange,
}: {
  root: TaskNodeInput
  problems: TaskProblem[]
  onChange: (next: TaskNodeInput) => void
}) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const patch = (key: string, delta: Partial<TaskNodeInput>) =>
    onChange(atPath(root, pathOf(key), (node) => ({ ...node, ...delta })))

  const addChild = (key: string) => {
    onChange(
      atPath(root, pathOf(key), (node) => ({
        ...node,
        children: [...node.children, newNode({ display_name: 'New question' })],
      })),
    )
    setCollapsed((state) => ({ ...state, [key]: false }))
  }

  const remove = (key: string) => onChange(dropAt(root, pathOf(key)))

  return (
    <ul className="ttree">
      <TreeNode
        node={root}
        nodeKey="0"
        depth={0}
        last
        problems={problems}
        collapsed={collapsed}
        onToggleCollapse={(key) => setCollapsed((state) => ({ ...state, [key]: !state[key] }))}
        onPatch={patch}
        onAddChild={addChild}
        onRemove={remove}
      />
    </ul>
  )
}

function TreeNode({
  node,
  nodeKey,
  depth,
  last,
  problems,
  collapsed,
  onToggleCollapse,
  onPatch,
  onAddChild,
  onRemove,
}: {
  node: TaskNodeInput
  nodeKey: string
  depth: number
  last: boolean
  problems: TaskProblem[]
  collapsed: Record<string, boolean>
  onToggleCollapse: (key: string) => void
  onPatch: (key: string, delta: Partial<TaskNodeInput>) => void
  onAddChild: (key: string) => void
  onRemove: (key: string) => void
}) {
  const isOpen = !collapsed[nodeKey]
  // The messages live in the body, which only exists while the node is open,
  // so a collapsed node describes nothing rather than a missing element.
  const ids = useId()
  const nameId = `${ids}-name`
  const nodeIdId = `${ids}-id`
  const listId = `${ids}-list`
  const classes = cleanList(node.class_labels)
  const outputs = cleanList(node.regression_outputs)
  const needsClasses = node.mode.endsWith('classification')
  const broken = problems.some((problem) => problem.nodeKey === nodeKey && problem.severity === 'error')

  const firstError = (field: string): TaskProblem | undefined =>
    problemsFor(problems, nodeField(nodeKey, field))[0]
  const listField = needsClasses ? 'class_labels' : 'regression_outputs'
  const nameError = firstError('display_name')
  const idError = firstError('node_id')
  const listError = firstError(listField)

  const rename = (display_name: string) => {
    // The id keeps following the name for as long as it still matches it.
    const follows = !node.node_id.trim() || node.node_id === slugify(node.display_name)
    onPatch(nodeKey, follows ? { display_name, node_id: slugify(display_name) } : { display_name })
  }

  const changeMode = (mode: NodeMode) => {
    onPatch(nodeKey, {
      mode,
      class_labels: mode === 'binary_classification' && classes.length !== 2 ? ['CASE', 'CONTROL'] : mode.endsWith('classification') ? node.class_labels : [],
      regression_outputs: mode.endsWith('regression') ? node.regression_outputs : [],
    })
  }

  const summary = [
    MODE_LABELS[node.mode],
    needsClasses ? `${classes.length} class${classes.length === 1 ? '' : 'es'}` : `${outputs.length} output${outputs.length === 1 ? '' : 's'}`,
    node.children.length > 0 ? `${node.children.length} child${node.children.length === 1 ? '' : 'ren'}` : '',
  ]
    .filter(Boolean)
    .join(', ')

  return (
    <li className={clsx('ttree__item', last && 'ttree__item--last')} data-depth={depth}>
      <div className={clsx('ttree__node', broken && 'ttree__node--broken')} data-open={isOpen}>
        <div className="ttree__head">
          <button
            type="button"
            className="ttree__chevron"
            aria-expanded={isOpen}
            aria-label={isOpen ? 'Collapse this node' : 'Expand this node'}
            onClick={() => onToggleCollapse(nodeKey)}
          >
            <ChevronRight size={14} />
          </button>

          <Input
            value={node.display_name}
            onChange={(event) => rename(event.target.value)}
            placeholder="What is this node asking?"
            invalid={Boolean(nameError)}
            aria-invalid={nameError ? true : undefined}
            aria-describedby={nameError && isOpen ? nameId : undefined}
            aria-label="Node name"
            className="grow"
            style={{ minWidth: 140 }}
          />

          <Select
            value={node.mode}
            onChange={(event) => changeMode(event.target.value as NodeMode)}
            aria-label="Node mode"
            style={{ flex: 'none', width: 178 }}
          >
            {MODE_ORDER.map((mode) => (
              <option key={mode} value={mode}>
                {MODE_LABELS[mode]}
              </option>
            ))}
          </Select>

          <Tooltip content="Add a question answered inside this one">
            <Button size="sm" variant="ghost" icon={<Plus size={13} />} onClick={() => onAddChild(nodeKey)}>
              Child
            </Button>
          </Tooltip>

          {depth > 0 && (
            <Tooltip content="Remove this node and everything under it">
              <Button
                size="sm"
                variant="ghost"
                iconOnly
                icon={<Trash2 size={13} />}
                aria-label={`Remove ${node.display_name || 'node'}`}
                onClick={() => onRemove(nodeKey)}
              />
            </Tooltip>
          )}
        </div>

        {!isOpen ? (
          <div className="ttree__collapsed">
            <span className="t-tiny muted truncate">{summary}</span>
            {broken && <Badge tone="critical">needs work</Badge>}
          </div>
        ) : (
          <div className="ttree__body">
            {nameError && (
              <span id={nameId} className="field__error">
                {nameError.message}
              </span>
            )}

            <div className="ttree__grid">
              <label className="ttree__field">
                <span className="t-micro muted">Node id</span>
                <Input
                  mono
                  value={node.node_id}
                  onChange={(event) => onPatch(nodeKey, { node_id: slugify(event.target.value) })}
                  placeholder="node_id"
                  invalid={Boolean(idError)}
                  aria-invalid={idError ? true : undefined}
                  aria-describedby={nodeIdId}
                />
                {idError ? (
                  <span id={nodeIdId} className="field__error">
                    {idError.message}
                  </span>
                ) : (
                  <span id={nodeIdId} className="t-micro faint">
                    Follows the name until you change it.
                  </span>
                )}
              </label>

              <label className="ttree__field">
                <span className="t-micro muted">
                  {needsClasses
                    ? node.mode === 'binary_classification'
                      ? 'Two class labels'
                      : 'Three or more class labels'
                    : node.mode === 'univariate_regression'
                      ? 'One output name'
                      : 'Two or more output names'}
                </span>
                {needsClasses ? (
                  <ChipList
                    values={node.class_labels}
                    onChange={(class_labels) => onPatch(nodeKey, { class_labels })}
                    placeholder="Add a label, then Enter"
                    invalid={Boolean(listError)}
                    aria-label="Class labels"
                    aria-describedby={listError ? listId : undefined}
                  />
                ) : (
                  <ChipList
                    values={node.regression_outputs}
                    onChange={(regression_outputs) => onPatch(nodeKey, { regression_outputs })}
                    placeholder="Add an output, then Enter"
                    invalid={Boolean(listError)}
                    aria-label="Output names"
                    aria-describedby={listError ? listId : undefined}
                  />
                )}
                {listError && (
                  <span id={listId} className="field__error">
                    {listError.message}
                  </span>
                )}
              </label>
            </div>

            {!needsClasses && outputs.length > 0 && (
              <div className="ttree__units">
                <span className="t-micro muted">Units, optional</span>
                {outputs.map((output) => (
                  <div key={output} className="row gap-2">
                    <span className="t-tiny mono truncate" style={{ width: 150, flex: 'none' }}>
                      {output}
                    </span>
                    <Input
                      value={node.unit_by_output[output] ?? ''}
                      placeholder="for example: points"
                      onChange={(event) =>
                        onPatch(nodeKey, {
                          unit_by_output: { ...node.unit_by_output, [output]: event.target.value },
                        })
                      }
                    />
                  </div>
                ))}
              </div>
            )}

            <div className="row between gap-3">
              <span className="stack" style={{ gap: 1 }}>
                <span className="t-tiny semibold">Required</span>
                <span className="t-micro muted">
                  {node.required
                    ? 'The run must answer this node.'
                    : 'The run may leave this node unanswered when the evidence does not support it.'}
                </span>
              </span>
              <Toggle
                checked={node.required}
                onChange={(required) => onPatch(nodeKey, { required })}
                label={`${node.display_name || 'Node'} required`}
              />
            </div>
          </div>
        )}
      </div>

      {isOpen && node.children.length > 0 && (
        <ul className="ttree__kids">
          {node.children.map((child, index) => (
            <TreeNode
              key={`${nodeKey}.${index}`}
              node={child}
              nodeKey={`${nodeKey}.${index}`}
              depth={depth + 1}
              last={index === node.children.length - 1}
              problems={problems}
              collapsed={collapsed}
              onToggleCollapse={onToggleCollapse}
              onPatch={onPatch}
              onAddChild={onAddChild}
              onRemove={onRemove}
            />
          ))}
        </ul>
      )}

      {isOpen && node.children.length === 0 && depth < 3 && (
        <button type="button" className="ttree__add" onClick={() => onAddChild(nodeKey)}>
          <CornerDownRight size={12} />
          Add a question under {node.display_name || 'this node'}
        </button>
      )}
    </li>
  )
}
