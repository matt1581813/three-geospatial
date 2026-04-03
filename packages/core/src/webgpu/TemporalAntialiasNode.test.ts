import { beforeAll, describe, expect, test, vi } from 'vitest'

type TemporalAntialiasNodeModule = typeof import('./TemporalAntialiasNode')

let biasTemporalAlphaWithCurrentFrameWeight: TemporalAntialiasNodeModule['biasTemporalAlphaWithCurrentFrameWeight']

beforeAll(async () => {
  vi.stubGlobal('GPUShaderStage', {
    VERTEX: 1,
    FRAGMENT: 2,
    COMPUTE: 4
  })
  vi.stubGlobal('GPUBufferUsage', {
    MAP_READ: 1,
    MAP_WRITE: 2,
    COPY_SRC: 4,
    COPY_DST: 8,
    INDEX: 16,
    VERTEX: 32,
    UNIFORM: 64,
    STORAGE: 128,
    INDIRECT: 256,
    QUERY_RESOLVE: 512
  })
  vi.stubGlobal('GPUTextureUsage', {
    COPY_SRC: 1,
    COPY_DST: 2,
    TEXTURE_BINDING: 4,
    STORAGE_BINDING: 8,
    RENDER_ATTACHMENT: 16
  })
  vi.stubGlobal('GPUMapMode', {
    READ: 1,
    WRITE: 2
  })
  vi.stubGlobal('GPUColorWrite', {
    RED: 1,
    GREEN: 2,
    BLUE: 4,
    ALPHA: 8,
    ALL: 15
  })

  ;({ biasTemporalAlphaWithCurrentFrameWeight } = await import(
    './TemporalAntialiasNode'
  ))
})

describe('TemporalAntialiasNode current-frame mask weighting', () => {
  test('keeps the base temporal alpha when the current frame mask is zero', () => {
    expect(biasTemporalAlphaWithCurrentFrameWeight(0.12, 0)).toBeCloseTo(0.12)
  })

  test('promotes the final temporal alpha to one when the current frame mask is one', () => {
    expect(biasTemporalAlphaWithCurrentFrameWeight(0.12, 1)).toBeCloseTo(1)
  })
})
