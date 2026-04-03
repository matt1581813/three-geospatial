import { describe, expect, test } from 'vitest'

import { CloudsTemporalState } from './CloudsTemporalState'

describe('CloudsTemporalState', () => {
  test('requests reset on first observation and only when revision changes', () => {
    const state = new CloudsTemporalState()

    expect(state.observe({ historyInvalidationRevision: 1 })).toBe(true)
    expect(state.historyResetRequested).toBe(true)
    expect(state.consumeHistoryReset()).toBe(true)
    expect(state.historyResetRequested).toBe(false)

    expect(state.observe({ historyInvalidationRevision: 1 })).toBe(false)
    expect(state.historyResetRequested).toBe(false)

    expect(state.observe({ historyInvalidationRevision: 2 })).toBe(true)
    expect(state.historyResetRequested).toBe(true)
  })

  test('supports manual reset requests without changing revision tracking', () => {
    const state = new CloudsTemporalState()

    state.observe({ historyInvalidationRevision: 3 })
    state.consumeHistoryReset()

    state.requestHistoryReset()

    expect(state.historyResetRequested).toBe(true)
    expect(state.consumeHistoryReset()).toBe(true)
    expect(state.observedHistoryInvalidationRevision).toBe(3)
  })
})
