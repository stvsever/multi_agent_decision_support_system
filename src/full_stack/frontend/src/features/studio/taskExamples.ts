/**
 * Ready-made trees.
 *
 * Each one is a shape phenotyping questions actually take: a case decision
 * with the follow-up questions that only make sense once the case is made.
 * They double as documentation of the import format.
 */

import type { TaskNodeInput } from '@/lib/types'

export interface TaskTreeExample {
  id: string
  title: string
  summary: string
  root: TaskNodeInput
}

const node = (seed: Partial<TaskNodeInput> & Pick<TaskNodeInput, 'node_id' | 'display_name' | 'mode'>): TaskNodeInput => ({
  class_labels: [],
  regression_outputs: [],
  unit_by_output: {},
  required: true,
  children: [],
  ...seed,
})

export const TASK_TREE_EXAMPLES: TaskTreeExample[] = [
  {
    id: 'case_subtype_severity',
    title: 'Case, then subtype and severity',
    summary:
      'A case decision at the root, with the subtype and the severity asked only in the context of that decision.',
    root: node({
      node_id: 'psychosis_present',
      display_name: 'Psychosis present',
      mode: 'binary_classification',
      class_labels: ['PSYCHOSIS', 'NON_PSYCHOSIS'],
      children: [
        node({
          node_id: 'psychosis_subtype',
          display_name: 'Psychosis subtype',
          mode: 'multiclass_classification',
          class_labels: ['SCHIZOPHRENIA', 'SCHIZOAFFECTIVE', 'AFFECTIVE_PSYCHOSIS'],
          required: false,
        }),
        node({
          node_id: 'symptom_severity',
          display_name: 'Symptom severity',
          mode: 'univariate_regression',
          regression_outputs: ['panss_total'],
          unit_by_output: { panss_total: 'points' },
          required: false,
        }),
      ],
    }),
  },
  {
    id: 'cognitive_profile',
    title: 'Cognitive profile, then impairment',
    summary: 'Three cognitive scores predicted together, with the clinical call that follows from them.',
    root: node({
      node_id: 'cognitive_profile',
      display_name: 'Cognitive profile',
      mode: 'multivariate_regression',
      regression_outputs: ['working_memory_z', 'processing_speed_z', 'verbal_recall_z'],
      unit_by_output: { working_memory_z: 'z', processing_speed_z: 'z', verbal_recall_z: 'z' },
      children: [
        node({
          node_id: 'impairment_present',
          display_name: 'Clinically significant impairment',
          mode: 'binary_classification',
          class_labels: ['IMPAIRED', 'UNIMPAIRED'],
        }),
      ],
    }),
  },
  {
    id: 'depression_course',
    title: 'Course, then burden and response',
    summary: 'A staged course at the root, with the illness burden and the treatment question underneath it.',
    root: node({
      node_id: 'depression_course',
      display_name: 'Depression course',
      mode: 'multiclass_classification',
      class_labels: ['NO_EPISODE', 'SINGLE_EPISODE', 'RECURRENT'],
      children: [
        node({
          node_id: 'episode_burden',
          display_name: 'Episode burden',
          mode: 'multivariate_regression',
          regression_outputs: ['episode_count', 'months_symptomatic'],
          unit_by_output: { episode_count: 'episodes', months_symptomatic: 'months' },
          required: false,
        }),
        node({
          node_id: 'treatment_resistance',
          display_name: 'Treatment resistance',
          mode: 'binary_classification',
          class_labels: ['RESISTANT', 'RESPONSIVE'],
          required: false,
        }),
      ],
    }),
  },
]

/** A fresh copy, so applying an example never aliases the constant above. */
export function cloneExample(example: TaskTreeExample): TaskNodeInput {
  return JSON.parse(JSON.stringify(example.root)) as TaskNodeInput
}
