import { Data3DTexture, Vector3 } from 'three'
import { beforeAll, describe, expect, test, vi } from 'vitest'

type CloudsNodeModule = typeof import('./CloudsNode')

let updateStbnSamplingParameters: CloudsNodeModule['updateStbnSamplingParameters']
let mapScreenCoordinateToLowResCoordinate: CloudsNodeModule['mapScreenCoordinateToLowResCoordinate']
let createCloudsTemporalResolveNode: CloudsNodeModule['createCloudsTemporalResolveNode']
let CLOUDS_TEMPORAL_ALPHA: CloudsNodeModule['CLOUDS_TEMPORAL_ALPHA']
let CLOUDS_VARIANCE_GAMMA: CloudsNodeModule['CLOUDS_VARIANCE_GAMMA']
let CLOUDS_VELOCITY_THRESHOLD: CloudsNodeModule['CLOUDS_VELOCITY_THRESHOLD']
let CLOUDS_DEPTH_ERROR: CloudsNodeModule['CLOUDS_DEPTH_ERROR']

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
    updateStbnSamplingParameters,
    mapScreenCoordinateToLowResCoordinate,
    createCloudsTemporalResolveNode,
    CLOUDS_TEMPORAL_ALPHA,
    CLOUDS_VARIANCE_GAMMA,
    CLOUDS_VELOCITY_THRESHOLD,
    CLOUDS_DEPTH_ERROR
  } = await import('./CloudsNode'))
})

describe('CloudsNode STBN sampling', () => {
  test('derives runtime STBN scale and animated slice from the input texture', () => {
    const texture = new Data3DTexture(new Uint8Array(4 * 8 * 6), 4, 8, 6)
    const scale = new Vector3()

    const layer = updateStbnSamplingParameters(texture, 13, scale)

    expect(scale.toArray()).toEqual([0.25, 0.125, 1 / 6])
    expect(layer).toBe(1)
  })

  test('supports freezing the STBN layer for debug comparisons', () => {
    const texture = new Data3DTexture(new Uint8Array(4 * 8 * 6), 4, 8, 6)
    const scale = new Vector3()

    const layer = updateStbnSamplingParameters(texture, 13, scale, 0)

    expect(scale.toArray()).toEqual([0.25, 0.125, 1 / 6])
    expect(layer).toBe(0)
  })

  test('maps resolve donor coordinates from the actual low-resolution texture size', () => {
    expect(mapScreenCoordinateToLowResCoordinate(500, 1000, 250)).toBe(125)
    expect(mapScreenCoordinateToLowResCoordinate(500, 1000, 125)).toBe(62)
    expect(mapScreenCoordinateToLowResCoordinate(999, 1000, 125)).toBe(124)
  })

  test('forwards a provided mask texture into the temporal resolve helper', () => {
    const sentinel = {
      sentinel: true,
      temporalAlpha: { value: 0 },
      varianceGamma: { value: 0 },
      velocityThreshold: { value: 0 },
      depthError: { value: 0 }
    } as const
    const createTemporalAntialias = vi.fn(() =>
      vi.fn(() => sentinel)
    ) as unknown as typeof import('@takram/three-geospatial/webgpu')['temporalAntialias']
    const owner = { projectionMatrix: null }
    const outputNode = { id: 'output' } as any
    const depthNode = { id: 'depth' } as any
    const velocityNode = { id: 'velocity' } as any
    const currentFrameMaskNode = { id: 'mask' } as any
    const camera = { isCamera: true } as any

    const resolveNode = createCloudsTemporalResolveNode(
      createTemporalAntialias,
      owner,
      outputNode,
      depthNode,
      velocityNode,
      camera,
      currentFrameMaskNode
    )

    expect(resolveNode).toBe(sentinel)
    expect(createTemporalAntialias).toHaveBeenCalledWith(owner)
    expect(createTemporalAntialias.mock.results[0]?.value).toHaveBeenCalledWith(
      outputNode,
      depthNode,
      velocityNode,
      camera,
      currentFrameMaskNode
    )
  })

  test('applies the fixed clouds temporal tuning to the generic resolve helper', () => {
    const resolveNode = {
      temporalAlpha: { value: 0 },
      varianceGamma: { value: 0 },
      velocityThreshold: { value: 0 },
      depthError: { value: 0 }
    }
    const createTemporalAntialias = vi.fn(() =>
      vi.fn(() => resolveNode)
    ) as unknown as typeof import('@takram/three-geospatial/webgpu')['temporalAntialias']
    const owner = { projectionMatrix: null }
    const outputNode = { id: 'output' } as any
    const depthNode = { id: 'depth' } as any
    const velocityNode = { id: 'velocity' } as any
    const currentFrameMaskNode = { id: 'mask' } as any
    const camera = { isCamera: true } as any

    const result = createCloudsTemporalResolveNode(
      createTemporalAntialias,
      owner,
      outputNode,
      depthNode,
      velocityNode,
      camera,
      currentFrameMaskNode
    )

    expect(result).toBe(resolveNode)
    expect(resolveNode.temporalAlpha.value).toBe(CLOUDS_TEMPORAL_ALPHA)
    expect(resolveNode.varianceGamma.value).toBe(CLOUDS_VARIANCE_GAMMA)
    expect(resolveNode.velocityThreshold.value).toBe(
      CLOUDS_VELOCITY_THRESHOLD
    )
    expect(resolveNode.depthError.value).toBe(CLOUDS_DEPTH_ERROR)
  })
})
