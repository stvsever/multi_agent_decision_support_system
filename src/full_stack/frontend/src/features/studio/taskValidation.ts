/**
 * What the engine will accept, checked while the form is being typed.
 *
 * The service rejects a malformed task with a validation error, so the same
 * contracts are mirrored here. Every problem carries the field it belongs to,
 * which is what lets the designer show it against the control that caused it
 * instead of piling everything into one list at the bottom.
 */

import type { NodeMode, PredictionType, TaskNodeInput, TaskSpecInput } from '@/lib/types'

export type Severity = 'error' | 'warning'

export interface TaskProblem {
  /** A flat field name, or `node:<key>:<field>` for a node in the tree. */
  field: string
  message: string
  severity: Severity
  nodeKey?: string
  nodeName?: string
}

export interface TaskRequirement {
  field: string
  /** Reads after "needs": "a comparator label". */
  label: string
  met: boolean
  note?: string
  /** What the step header says instead of the label: "2 more classes". */
  short?: string
}

export interface TaskReport {
  ready: boolean
  problems: TaskProblem[]
  requirements: TaskRequirement[]
  /** One line for the step header: what is still missing, or what will run. */
  summary: string
  nodeCount: number
}

export const MODE_LABELS: Record<NodeMode, string> = {
  binary_classification: 'Binary',
  multiclass_classification: 'Multiclass',
  univariate_regression: 'Univariate regression',
  multivariate_regression: 'Multivariate regression',
}

export const MODE_ORDER: NodeMode[] = [
  'binary_classification',
  'multiclass_classification',
  'univariate_regression',
  'multivariate_regression',
]

export const FAMILY_LABELS: Record<PredictionType, string> = {
  binary: 'Binary classification',
  multiclass: 'Multiclass classification',
  regression_univariate: 'Univariate regression',
  regression_multivariate: 'Multivariate regression',
  hierarchical: 'Task tree',
}

/** The engine trims, drops blanks, and de-duplicates before it counts. */
export function cleanList(values: readonly string[] | undefined | null): string[] {
  const out: string[] = []
  for (const raw of values ?? []) {
    const text = String(raw ?? '').trim()
    if (text && !out.includes(text)) out.push(text)
  }
  return out
}

export const nodeField = (key: string, field: string): string => `node:${key}:${field}`

/** Node ids follow the display name until someone edits them by hand. */
export function slugify(text: string): string {
  return String(text ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 60)
}

export function walkNodes(
  node: TaskNodeInput | null | undefined,
  visit: (node: TaskNodeInput, key: string, depth: number) => void,
  key = '0',
  depth = 0,
): void {
  if (!node) return
  visit(node, key, depth)
  node.children?.forEach((child, index) => walkNodes(child, visit, `${key}.${index}`, depth + 1))
}

export function countNodes(node: TaskNodeInput | null | undefined): number {
  let total = 0
  walkNodes(node, () => {
    total += 1
  })
  return total
}

/* --- Node contracts -------------------------------------------------------- */

function modeProblems(node: TaskNodeInput, key: string, name: string): TaskProblem[] {
  const classes = cleanList(node.class_labels)
  const outputs = cleanList(node.regression_outputs)
  const problems: TaskProblem[] = []
  const push = (field: string, message: string, severity: Severity = 'error') =>
    problems.push({ field: nodeField(key, field), message, severity, nodeKey: key, nodeName: name })

  if (node.mode === 'binary_classification') {
    if (classes.length < 2) {
      push(
        'class_labels',
        classes.length === 0
          ? 'Two distinct class labels are needed, one for each side of the comparison.'
          : 'One more class label is needed. Binary compares exactly two.',
      )
    } else if (classes.length > 2) {
      push('class_labels', `Binary takes exactly two class labels. Remove ${classes.length - 2}.`)
    }
    if (outputs.length > 0) push('regression_outputs', 'A classification node cannot also predict numbers.')
  }

  if (node.mode === 'multiclass_classification') {
    if (classes.length < 3) {
      const missing = 3 - classes.length
      push(
        'class_labels',
        classes.length === 0
          ? 'Three or more mutually exclusive classes are needed.'
          : `${missing} more class label${missing === 1 ? '' : 's'} needed, three at minimum.`,
      )
    }
    if (outputs.length > 0) push('regression_outputs', 'A classification node cannot also predict numbers.')
  }

  if (node.mode === 'univariate_regression') {
    if (outputs.length === 0) push('regression_outputs', 'One continuous output name is needed.')
    else if (outputs.length > 1) {
      push('regression_outputs', `Univariate predicts one value. Remove ${outputs.length - 1}, or switch to multivariate.`)
    }
    if (classes.length > 0) push('class_labels', 'A regression node cannot also define class labels.')
  }

  if (node.mode === 'multivariate_regression') {
    if (outputs.length < 2) {
      push(
        'regression_outputs',
        outputs.length === 0
          ? 'Two or more continuous outputs are needed.'
          : 'One more output is needed. Multivariate predicts two or more together.',
      )
    }
    if (classes.length > 0) push('class_labels', 'A regression node cannot also define class labels.')
  }

  return problems
}

function treeProblems(root: TaskNodeInput): TaskProblem[] {
  const problems: TaskProblem[] = []
  const idCount = new Map<string, number>()

  walkNodes(root, (node) => {
    const id = node.node_id.trim()
    if (id) idCount.set(id, (idCount.get(id) ?? 0) + 1)
  })

  walkNodes(root, (node, key) => {
    const name = node.display_name.trim() || node.node_id.trim() || 'this node'
    const id = node.node_id.trim()

    if (!node.display_name.trim()) {
      problems.push({
        field: nodeField(key, 'display_name'),
        message: 'Every node needs a name. It is what the agents are asked about.',
        severity: 'error',
        nodeKey: key,
        nodeName: name,
      })
    }
    if (!id) {
      problems.push({
        field: nodeField(key, 'node_id'),
        message: 'Every node needs an id. It keys the answer in the result.',
        severity: 'error',
        nodeKey: key,
        nodeName: name,
      })
    } else if ((idCount.get(id) ?? 0) > 1) {
      problems.push({
        field: nodeField(key, 'node_id'),
        // The engine indexes answers by id, so a repeat quietly overwrites one.
        message: `The id "${id}" is used by more than one node. Only one answer would survive.`,
        severity: 'error',
        nodeKey: key,
        nodeName: name,
      })
    }

    problems.push(...modeProblems(node, key, name))
  })

  return problems
}

/* --- Whole task ------------------------------------------------------------ */

export function validateTask(task: TaskSpecInput): TaskProblem[] {
  const problems: TaskProblem[] = []
  const target = task.target_label.trim()
  const control = task.control_label.trim()
  const classes = cleanList(task.class_labels)
  const outputs = cleanList(task.regression_outputs)

  if (task.prediction_type === 'hierarchical') {
    if (!task.root) {
      problems.push({ field: 'root', message: 'A task tree needs a root question.', severity: 'error' })
      return problems
    }
    return treeProblems(task.root)
  }

  if (!target) {
    problems.push({
      field: 'target_label',
      message: 'Name the phenotype being predicted. This name reaches every agent prompt.',
      severity: 'error',
    })
  }

  if (task.prediction_type === 'binary') {
    if (!control) {
      problems.push({
        field: 'control_label',
        message: 'Name what the target is being distinguished from.',
        severity: 'error',
      })
    } else if (control.toLowerCase() === target.toLowerCase()) {
      problems.push({
        field: 'control_label',
        message: 'The comparator has to differ from the target, otherwise there is only one class.',
        severity: 'error',
      })
    }
    // The engine prefers two stored class labels over the two names above, so
    // a leftover pair from another family would quietly decide the classes.
    if (classes.length === 2 && (classes[0] !== target || classes[1] !== control)) {
      problems.push({
        field: 'control_label',
        message: `This task also carries the class labels ${classes.join(' and ')}, and binary sends those to the engine in place of the two names above. Pick the family again to clear them.`,
        severity: 'warning',
      })
    }
  }

  if (task.prediction_type === 'multiclass') {
    if (classes.length < 3) {
      const missing = 3 - classes.length
      problems.push({
        field: 'class_labels',
        message:
          classes.length === 0
            ? 'Add three or more mutually exclusive classes.'
            : `Add ${missing} more class${missing === 1 ? '' : 'es'}. Multiclass needs three at minimum.`,
        severity: 'error',
      })
    }
  }

  if (task.prediction_type === 'regression_univariate') {
    if (outputs.length === 0) {
      problems.push({ field: 'regression_outputs', message: 'Name the one continuous output.', severity: 'error' })
    } else if (outputs.length > 1) {
      problems.push({
        field: 'regression_outputs',
        message: `Univariate predicts a single value. Remove ${outputs.length - 1}, or switch to multivariate.`,
        severity: 'error',
      })
    }
  }

  if (task.prediction_type === 'regression_multivariate') {
    if (outputs.length < 2) {
      problems.push({
        field: 'regression_outputs',
        message:
          outputs.length === 0
            ? 'Name two or more continuous outputs.'
            : 'Add one more output. Multivariate predicts two or more together.',
        severity: 'error',
      })
    }
  }

  return problems
}

function requirementsFor(task: TaskSpecInput, problems: TaskProblem[]): TaskRequirement[] {
  const target = task.target_label.trim()
  const control = task.control_label.trim()
  const classes = cleanList(task.class_labels)
  const outputs = cleanList(task.regression_outputs)
  const named: TaskRequirement = { field: 'target_label', label: 'a target name', met: Boolean(target) }
  const more = (count: number, one: string, many: string) => `${count} more ${count === 1 ? one : many}`

  switch (task.prediction_type) {
    case 'binary':
      return [
        named,
        {
          field: 'control_label',
          label: 'a comparator label',
          met: Boolean(control) && control.toLowerCase() !== target.toLowerCase(),
          note: control && control.toLowerCase() === target.toLowerCase() ? 'same as the target' : undefined,
          short:
            control && control.toLowerCase() === target.toLowerCase()
              ? 'a comparator that differs from the target'
              : 'a comparator label',
        },
      ]
    case 'multiclass':
      return [
        named,
        {
          field: 'class_labels',
          label: 'three or more classes',
          met: classes.length >= 3,
          note: `${classes.length} of 3`,
          short: classes.length === 0 ? 'three class labels' : more(3 - classes.length, 'class', 'classes'),
        },
      ]
    case 'regression_univariate':
      return [
        named,
        {
          field: 'regression_outputs',
          label: 'exactly one output',
          met: outputs.length === 1,
          note: outputs.length === 1 ? undefined : `${outputs.length} set`,
          short:
            outputs.length === 0
              ? 'an output name'
              : `one output only, so remove ${outputs.length - 1}`,
        },
      ]
    case 'regression_multivariate':
      return [
        named,
        {
          field: 'regression_outputs',
          label: 'two or more outputs',
          met: outputs.length >= 2,
          note: `${outputs.length} of 2`,
          short: outputs.length === 0 ? 'two output names' : more(2 - outputs.length, 'output', 'outputs'),
        },
      ]
    default: {
      const total = countNodes(task.root)
      const broken = new Set(problems.filter((p) => p.severity === 'error' && p.nodeKey).map((p) => p.nodeKey))
      return [
        { field: 'root', label: 'a root question', met: Boolean(task.root) },
        {
          field: 'tree',
          label: 'every node complete',
          met: Boolean(task.root) && broken.size === 0,
          note: total > 0 ? `${total - broken.size} of ${total} node${total === 1 ? '' : 's'}` : undefined,
        },
      ]
    }
  }
}

function joinNeeds(labels: string[]): string {
  if (labels.length === 0) return ''
  if (labels.length === 1) return labels[0]
  return `${labels.slice(0, -1).join(', ')} and ${labels[labels.length - 1]}`
}

function readySummary(task: TaskSpecInput, nodeCount: number): string {
  const family = FAMILY_LABELS[task.prediction_type]
  const target = task.target_label.trim()
  const classes = cleanList(task.class_labels)
  const outputs = cleanList(task.regression_outputs)

  switch (task.prediction_type) {
    case 'binary':
      return `${family}: ${target} against ${task.control_label.trim()}`
    case 'multiclass':
      return `${family}: ${classes.length} classes, ${classes.slice(0, 3).join(', ')}${classes.length > 3 ? ' and more' : ''}`
    case 'regression_univariate':
      return `${family}: ${outputs[0] ?? target}`
    case 'regression_multivariate':
      return `${family}: ${outputs.length} outputs, ${outputs.slice(0, 3).join(', ')}${outputs.length > 3 ? ' and more' : ''}`
    default:
      return `${family}: ${nodeCount} node${nodeCount === 1 ? '' : 's'}, root is ${task.root?.display_name?.trim() || 'unnamed'}`
  }
}

/**
 * The whole picture in one call: what is wrong, what is still needed, and the
 * single line the step header shows. The line is written per family, because
 * "two things to fix" says nothing about which family you are filling in.
 */
export function taskReport(task: TaskSpecInput): TaskReport {
  const problems = validateTask(task)
  const requirements = requirementsFor(task, problems)
  const nodeCount = countNodes(task.root)
  const errors = problems.filter((problem) => problem.severity === 'error')
  const ready = errors.length === 0

  if (ready) return { ready, problems, requirements, summary: readySummary(task, nodeCount), nodeCount }

  if (task.prediction_type === 'hierarchical') {
    if (!task.root) return { ready, problems, requirements, summary: 'needs a root question', nodeCount }
    const first = errors[0]
    const others = new Set(errors.map((problem) => problem.nodeKey)).size - 1
    const where = first.nodeName ? `${first.nodeName}: ` : ''
    const tail = others > 0 ? ` (and ${others} other node${others === 1 ? '' : 's'})` : ''
    return {
      ready,
      problems,
      requirements,
      summary: `${where}${lowerFirst(first.message)}${tail}`,
      nodeCount,
    }
  }

  const unmet = requirements.filter((requirement) => !requirement.met)
  if (unmet.length > 0) {
    return {
      ready,
      problems,
      requirements,
      summary: `needs ${joinNeeds(unmet.map((requirement) => requirement.short ?? requirement.label))}`,
      nodeCount,
    }
  }
  return { ready, problems, requirements, summary: lowerFirst(errors[0].message), nodeCount }
}

function lowerFirst(text: string): string {
  return text.charAt(0).toLowerCase() + text.slice(1)
}

/** Problems attached to one control, in severity order. */
export function problemsFor(problems: TaskProblem[], field: string): TaskProblem[] {
  return problems
    .filter((problem) => problem.field === field)
    .sort((a, b) => (a.severity === b.severity ? 0 : a.severity === 'error' ? -1 : 1))
}
