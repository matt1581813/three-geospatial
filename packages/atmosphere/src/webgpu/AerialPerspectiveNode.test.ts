import { beforeAll, describe, expect, test, vi } from 'vitest'

type AerialPerspectiveModule = typeof import('./AerialPerspectiveNode')

let resolveCloudsAerialProviders: AerialPerspectiveModule['resolveCloudsAerialProviders']

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

  ;({ resolveCloudsAerialProviders } = await import('./AerialPerspectiveNode'))
})

describe('AerialPerspective cloud providers', () => {
  test('returns null providers when builder context has no cloud hooks', () => {
    const result = resolveCloudsAerialProviders(
      { context: {} } as any,
      { id: 'uv' } as any
    )
    expect(result.sampleCloudShadow).toBeNull()
    expect(result.shadowLengthNode).toBeNull()
  })

  test('resolves cloud-shadow sampler from getCloudsShadow()', () => {
    const sample = vi.fn(() => 0.7)
    const result = resolveCloudsAerialProviders(
      {
        context: {
          getCloudsShadow: () => ({ sample })
        }
      } as any,
      { id: 'uv' } as any
    )
    expect(result.sampleCloudShadow).toBe(sample)
  })

  test('resolves cloud shadow-length node from sampleShadowLength()', () => {
    const sampleShadowLength = vi.fn(() => ({ id: 'shadow-length' }))
    const result = resolveCloudsAerialProviders(
      {
        context: {
          getCloudsShadowLength: () => ({ sampleShadowLength })
        }
      } as any,
      { id: 'uv' } as any
    )
    expect(sampleShadowLength).toHaveBeenCalledTimes(1)
    expect(result.shadowLengthNode).toEqual({ id: 'shadow-length' })
  })

  test('resolves cloud shadow-length node from getTextureNode().sample().r', () => {
    const sample = vi.fn(() => ({ r: { id: 'shadow-length-from-texture' } }))
    const result = resolveCloudsAerialProviders(
      {
        context: {
          getCloudsShadowLength: () => ({
            getTextureNode: () => ({ sample })
          })
        }
      } as any,
      { id: 'uv' } as any
    )
    expect(sample).toHaveBeenCalledTimes(1)
    expect(result.shadowLengthNode).toEqual({
      id: 'shadow-length-from-texture'
    })
  })
})

