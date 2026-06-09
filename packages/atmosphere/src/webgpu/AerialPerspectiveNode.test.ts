import { beforeAll, describe, expect, test, vi } from 'vitest'

type AerialPerspectiveModule = typeof import('./AerialPerspectiveNode')
type TslModule = typeof import('three/tsl')

let resolveCloudsAerialProviders: AerialPerspectiveModule['resolveCloudsAerialProviders']
let float: TslModule['float']

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

  ;({ float } = await import('three/tsl'))
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
    const positionWorldNode = { id: 'position' } as any
    const normalWorldNode = { id: 'normal' } as any
    const distancePositionUnitNode = { id: 'distance-position' } as any
    const provider = {
      marker: 'cloud-shadow-provider',
      sample: vi.fn(function (
        this: { marker: string },
        positionWorldNode: unknown,
        normalWorldNode: unknown,
        distancePositionUnitNode: unknown
      ) {
        return {
          marker: this.marker,
          positionWorldNode,
          normalWorldNode,
          distancePositionUnitNode
        }
      })
    }
    const result = resolveCloudsAerialProviders(
      {
        context: {
          getCloudsShadow: () => provider
        }
      } as any,
      { id: 'uv' } as any
    )
    expect(result.sampleCloudShadow).not.toBe(provider.sample)
    expect(
      result.sampleCloudShadow?.(
        positionWorldNode,
        normalWorldNode,
        distancePositionUnitNode
      )
    ).toEqual({
      marker: 'cloud-shadow-provider',
      positionWorldNode,
      normalWorldNode,
      distancePositionUnitNode
    })
    expect(provider.sample).toHaveBeenCalledWith(
      positionWorldNode,
      normalWorldNode,
      distancePositionUnitNode
    )
  })

  test('resolves cloud shadow-length node from sampleShadowLength()', () => {
    const sampleShadowLength = vi.fn(() => float(0.7))
    const uvNode = { id: 'uv' } as any
    const result = resolveCloudsAerialProviders(
      {
        context: {
          getCloudsShadowLength: () => ({ sampleShadowLength })
        }
      } as any,
      uvNode
    )
    expect(sampleShadowLength).toHaveBeenCalledTimes(1)
    expect(sampleShadowLength).toHaveBeenCalledWith(uvNode)
    expect(result.shadowLengthNode).not.toBeNull()
  })

  test('resolves cloud shadow-length node from getTextureNode().sample().r', () => {
    const sample = vi.fn(() => ({ r: float(0.5) }))
    const uvNode = { id: 'uv' } as any
    const result = resolveCloudsAerialProviders(
      {
        context: {
          getCloudsShadowLength: () => ({
            getTextureNode: () => ({ sample })
          })
        }
      } as any,
      uvNode
    )
    expect(sample).toHaveBeenCalledTimes(1)
    expect(sample).toHaveBeenCalledWith(uvNode)
    expect(result.shadowLengthNode).not.toBeNull()
  })
})

