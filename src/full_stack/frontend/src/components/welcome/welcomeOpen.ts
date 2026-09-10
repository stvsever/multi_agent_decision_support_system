/**
 * Whether the welcome is being shown again on request.
 *
 * The first-run marker in the app store is persisted, so it cannot double as
 * open/closed state: reopening About by clearing it wrote "never seen" back to
 * local storage and the dialog then appeared unbidden on the next reload. This
 * is the transient half of that state and it is deliberately not persisted and
 * not part of the store, which is frozen for this change.
 */

import { useSyncExternalStore } from 'react'

let reopened = false
const listeners = new Set<() => void>()

const emit = () => {
  for (const listener of listeners) listener()
}

/** Show the welcome again without touching the persisted first-run marker. */
export function openWelcome(): void {
  if (reopened) return
  reopened = true
  emit()
}

export function closeWelcome(): void {
  if (!reopened) return
  reopened = false
  emit()
}

export function useWelcomeReopened(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      listeners.add(onChange)
      return () => {
        listeners.delete(onChange)
      }
    },
    () => reopened,
    () => false,
  )
}
