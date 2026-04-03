import { beforeAll, describe, expect, test, vi } from 'vitest'

type CloudsShadowLengthModule = typeof import('./CloudsShadowLengthNode')

let clampShadowLengthDistance: CloudsShadowLengthModule['clampShadowLengthDistance']
let computeShadowLengthContribution: CloudsShadowLengthModule['computeShadowLengthContribution']
let SHADOW_LENGTH_TEMPORAL_ALPHA: CloudsShadowLengthModule['SHADOW_LENGTH_TEMPORAL_ALPHA']
let SHADOW_LENGTH_VARIANCE_GAMMA: CloudsShadowLengthModule['SHADOW_LENGTH_VARIANCE_GAMMA']

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
  ;({
    clampShadowLengthDistance,
    computeShadowLengthContribution,
    SHADOW_LENGTH_TEMPORAL_ALPHA,
    SHADOW_LENGTH_VARIANCE_GAMMA
  } = await import('./CloudsShadowLengthNode'))
})

describe('CloudsShadowLengthNode helpers', () => {
  test('clamps scene distance by max shadow-length distance', () => {
    expect(clampShadowLengthDistance(1200, 1000)).toBe(1000)
    expect(clampShadowLengthDistance(800, 1000)).toBe(800)
    expect(clampShadowLengthDistance(-1, 1000)).toBe(0)
  })

  test('matches Beer-style per-step shadow-length accumulation term', () => {
    expect(computeShadowLengthContribution(0, 100)).toBe(0)
    expect(computeShadowLengthContribution(1, 100)).toBeCloseTo(
      (1 - Math.exp(-1)) * 100
    )
    expect(computeShadowLengthContribution(2, 40, 0.5)).toBeCloseTo(
      (1 - Math.exp(-2)) * 40 * 0.5
    )
  })

  test('uses the WebGL shadow-length temporal defaults for resolve history', () => {
    expect(SHADOW_LENGTH_TEMPORAL_ALPHA).toBe(0.1)
    expect(SHADOW_LENGTH_VARIANCE_GAMMA).toBe(2)
  })
})
