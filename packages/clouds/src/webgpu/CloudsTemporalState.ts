export interface CloudsTemporalHistorySource {
  historyInvalidationRevision: number
}

export class CloudsTemporalState {
  private observedRevision = -1
  private resetRequested = true

  observe(source: CloudsTemporalHistorySource): boolean {
    const nextRevision = source.historyInvalidationRevision
    if (this.observedRevision === nextRevision) {
      return false
    }
    this.observedRevision = nextRevision
    this.resetRequested = true
    return true
  }

  get historyResetRequested(): boolean {
    return this.resetRequested
  }

  consumeHistoryReset(): boolean {
    const value = this.resetRequested
    this.resetRequested = false
    return value
  }

  requestHistoryReset(): void {
    this.resetRequested = true
  }

  get observedHistoryInvalidationRevision(): number {
    return this.observedRevision
  }
}
