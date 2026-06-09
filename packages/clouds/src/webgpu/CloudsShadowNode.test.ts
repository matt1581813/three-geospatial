import { Vector2 } from 'three'
import { beforeAll, describe, expect, test, vi } from 'vitest'

type CloudsShadowModule = typeof import('./CloudsShadowNode')

let clampCloudShadowCascadeCount: CloudsShadowModule['clampCloudShadowCascadeCount']
let getCloudShadowAtlasSize: CloudsShadowModule['getCloudShadowAtlasSize']
let getCloudShadowAtlasViewport: CloudsShadowModule['getCloudShadowAtlasViewport']
let getCloudShadowCascadeDepth: CloudsShadowModule['getCloudShadowCascadeDepth']
let getCloudShadowCascadeFadeWidth: CloudsShadowModule['getCloudShadowCascadeFadeWidth']
let getCloudShadowCascadeBlendWeights: CloudsShadowModule['getCloudShadowCascadeBlendWeights']
let resolveCloudShadowCascadeIndex: CloudsShadowModule['resolveCloudShadowCascadeIndex']
let resolveCloudShadowFadedCascadeIndex: CloudsShadowModule['resolveCloudShadowFadedCascadeIndex']
let computeCloudShadowOpticalDepth: CloudsShadowModule['computeCloudShadowOpticalDepth']
let computeCloudShadowSurfaceOpticalDepth: CloudsShadowModule['computeCloudShadowSurfaceOpticalDepth']
let computeCloudShadowOpticalDepthTail: CloudsShadowModule['computeCloudShadowOpticalDepthTail']
let computeCloudShadowFilterRadius: CloudsShadowModule['computeCloudShadowFilterRadius']
let WEBGPU_CLOUD_BODY_MAX_FILTER_RADIUS: CloudsShadowModule['WEBGPU_CLOUD_BODY_MAX_FILTER_RADIUS']

beforeAll(() => {
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
})

beforeAll(async () => {
  ;({
    clampCloudShadowCascadeCount,
    getCloudShadowAtlasSize,
    getCloudShadowAtlasViewport,
    getCloudShadowCascadeDepth,
    getCloudShadowCascadeFadeWidth,
    getCloudShadowCascadeBlendWeights,
    resolveCloudShadowCascadeIndex,
    resolveCloudShadowFadedCascadeIndex,
    computeCloudShadowOpticalDepth,
    computeCloudShadowSurfaceOpticalDepth,
    computeCloudShadowOpticalDepthTail,
    computeCloudShadowFilterRadius,
    WEBGPU_CLOUD_BODY_MAX_FILTER_RADIUS
  } = await import('./CloudsShadowNode'))
})

describe('CloudsShadowNode helpers', () => {
  test('clamps cascade count to the supported atlas range', () => {
    expect(clampCloudShadowCascadeCount(0)).toBe(1)
    expect(clampCloudShadowCascadeCount(2.2)).toBe(2)
    expect(clampCloudShadowCascadeCount(8)).toBe(4)
  })

  test('computes atlas size and tile viewport from cascade count', () => {
    const mapSize = new Vector2(256, 192)

    expect(getCloudShadowAtlasSize(mapSize, 3).toArray()).toEqual([768, 192])
    expect(getCloudShadowAtlasViewport(0, mapSize, 3)).toEqual({
      x: 0,
      y: 0,
      width: 256,
      height: 192
    })
    expect(getCloudShadowAtlasViewport(2, mapSize, 3)).toEqual({
      x: 512,
      y: 0,
      width: 256,
      height: 192
    })
  })

  test('converts view depth to normalized cascade depth', () => {
    expect(getCloudShadowCascadeDepth(0.1, 0.1, 10_000)).toBeCloseTo(0)
    expect(getCloudShadowCascadeDepth(5_000.05, 0.1, 10_000)).toBeCloseTo(0.5)
    expect(getCloudShadowCascadeDepth(10_000, 0.1, 10_000)).toBeCloseTo(1)
  })

  test('uses bounded fade widths around cascade boundaries', () => {
    expect(getCloudShadowCascadeFadeWidth(0.01)).toBeCloseTo(0.004)
    expect(getCloudShadowCascadeFadeWidth(0.2)).toBeCloseTo(0.02)
    expect(getCloudShadowCascadeFadeWidth(1)).toBeCloseTo(0.06)
  })

  test('selects the last matching cascade interval for normalized receiver depth', () => {
    const intervals = [
      new Vector2(0, 0.13339799558860332),
      new Vector2(0.13339799558860332, 0.2728812798267225),
      new Vector2(0.2728812798267225, 1)
    ]

    expect(resolveCloudShadowCascadeIndex(0.05, intervals, 3)).toBe(0)
    expect(resolveCloudShadowCascadeIndex(0.2, intervals, 3)).toBe(1)
    expect(resolveCloudShadowCascadeIndex(0.6, intervals, 3)).toBe(2)
  })

  test('matches WebGL stochastic cascade fade selection for Beer shadow samples', () => {
    const intervals = [
      new Vector2(0, 0.13339799558860332),
      new Vector2(0.13339799558860332, 0.2728812798267225),
      new Vector2(0.2728812798267225, 1)
    ]
    const boundary = intervals[0].y
    const fadeWidth = boundary * boundary * 0.5
    const expandedStart = boundary - fadeWidth * 0.5
    const quarterFadeDepth = expandedStart + fadeWidth * 0.25

    expect(
      resolveCloudShadowFadedCascadeIndex(0.05, intervals, 3, 0.75)
    ).toBe(0)
    expect(
      resolveCloudShadowFadedCascadeIndex(
        quarterFadeDepth,
        intervals,
        3,
        0.2
      )
    ).toBe(1)
    expect(
      resolveCloudShadowFadedCascadeIndex(
        quarterFadeDepth,
        intervals,
        3,
        0.8
      )
    ).toBe(0)
    expect(
      resolveCloudShadowFadedCascadeIndex(0.6, intervals, 3, 0.75)
    ).toBe(2)
  })

  test('produces stable normalized blend weights across cascade transitions', () => {
    const intervals = [
      new Vector2(0, 0.13339799558860332),
      new Vector2(0.13339799558860332, 0.2728812798267225),
      new Vector2(0.2728812798267225, 1)
    ]

    expect(getCloudShadowCascadeBlendWeights(0.05, intervals, 3)).toEqual([
      1, 0, 0, 0
    ])

    const edgeWeights = getCloudShadowCascadeBlendWeights(
      intervals[0].y,
      intervals,
      3
    )
    expect(edgeWeights[0]).toBeCloseTo(0.5, 5)
    expect(edgeWeights[1]).toBeCloseTo(0.5, 5)
    expect(edgeWeights[2]).toBeCloseTo(0, 5)
    expect(edgeWeights[3]).toBeCloseTo(0, 5)

    const farWeights = getCloudShadowCascadeBlendWeights(0.7, intervals, 3)
    expect(farWeights[0]).toBeCloseTo(0, 5)
    expect(farWeights[1]).toBeCloseTo(0, 5)
    expect(farWeights[2]).toBeCloseTo(1, 5)
    expect(farWeights[3]).toBeCloseTo(0, 5)
  })

  test('reconstructs Beer optical depth from packed shadow atlas data', () => {
    expect(
      computeCloudShadowOpticalDepth(18_000, 2_000, 9_000, 0.0015, 11, 0.75)
    ).toBeCloseTo(10.5)
    expect(
      computeCloudShadowOpticalDepth(12_000, 0, 20_000, 0.002, 7, 1)
    ).toBeCloseTo(0)
  })

  test('matches WebGL surface shadow reconstruction without optical-depth tail', () => {
    expect(
      computeCloudShadowSurfaceOpticalDepth(18_000, 9_000, 0.0015, 11)
    ).toBeCloseTo(11)
    expect(
      computeCloudShadowSurfaceOpticalDepth(18_000, 9_000, 0.0015, 17)
    ).toBeCloseTo(13.5)
    expect(
      computeCloudShadowSurfaceOpticalDepth(12_000, 20_000, 0.002, 7)
    ).toBeCloseTo(0)
  })

  test('matches the WebGL optical-depth tail heuristic', () => {
    expect(computeCloudShadowOpticalDepthTail(100, 1, 2)).toBeCloseTo(50)
    expect(computeCloudShadowOpticalDepthTail(100, 6, 2)).toBeCloseTo(
      2 * 100 * Math.exp(-5)
    )
  })

  test('matches the WebGL horizon-weighted Beer shadow filter radius', () => {
    expect(
      computeCloudShadowFilterRadius(0.2, WEBGPU_CLOUD_BODY_MAX_FILTER_RADIUS)
    ).toBeCloseTo(0)
    expect(
      computeCloudShadowFilterRadius(0.05, WEBGPU_CLOUD_BODY_MAX_FILTER_RADIUS)
    ).toBeCloseTo(WEBGPU_CLOUD_BODY_MAX_FILTER_RADIUS * 0.5)
    expect(
      computeCloudShadowFilterRadius(0, WEBGPU_CLOUD_BODY_MAX_FILTER_RADIUS)
    ).toBeCloseTo(WEBGPU_CLOUD_BODY_MAX_FILTER_RADIUS)
  })
})
