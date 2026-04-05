import { Data3DTexture, Texture, Vector2, Vector3 } from 'three'
import type { NodeBuilder } from 'three/webgpu'
import { beforeAll, describe, expect, test, vi } from 'vitest'

import {
  WEBGPU_MAX_PRIMARY_STEPS,
  WEBGPU_MAX_SECONDARY_STEPS,
  webgpuQualityPresets
} from './qualityPresets'

type CloudsModule = typeof import('./CloudsContext')

let CloudsContext: CloudsModule['CloudsContext']
let getCloudsContext: CloudsModule['getCloudsContext']

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
  ;({ CloudsContext, getCloudsContext } = await import('./CloudsContext'))
})

describe('CloudsContext', () => {
  test('initializes with expected defaults and fallback textures', () => {
    const context = new CloudsContext()

    expect(context.temporalUpscale).toBe(false)
    expect(context.temporalAntialias).toBe(true)
    expect(context.historyInvalidationRevision).toBe(1)
    expect(context.qualityPreset).toBe('high')
    expect(context.resolutionScale).toBe(
      webgpuQualityPresets.high.resolutionScale
    )
    expect(context.temporalUpscaleScale).toBe(0.375)
    expect(context.lightShafts).toBe(webgpuQualityPresets.high.lightShafts)
    expect(context.shapeDetail).toBe(webgpuQualityPresets.high.shapeDetail)
    expect(context.turbulence).toBe(webgpuQualityPresets.high.turbulence)
    expect(context.haze).toBe(webgpuQualityPresets.high.haze)
    expect(context.shadow.cascadeCount).toBe(
      webgpuQualityPresets.high.shadow.cascadeCount
    )
    expect(context.shadow.mapSize.toArray()).toEqual([
      webgpuQualityPresets.high.shadow.mapSize,
      webgpuQualityPresets.high.shadow.mapSize
    ])
    expect(context.shadow.splitLambda).toBe(
      webgpuQualityPresets.high.shadow.splitLambda
    )
    expect(context.animateStbn).toBe(true)
    expect(context.stbnFrameIndex).toBe(0)
    expect(context.useStbnNode.value).toBe(false)
    expect(context.resolvedLocalWeatherTexture).toBeDefined()
    expect(context.resolvedShapeTexture).toBeDefined()
    expect(context.resolvedShapeDetailTexture).toBeDefined()
    expect(context.resolvedTurbulenceTexture).toBeDefined()
  })

  test('applies quality preset values and syncs nodes', () => {
    const context = new CloudsContext()
    const revision = context.historyInvalidationRevision

    context.qualityPreset = 'low'

    expect(context.historyInvalidationRevision).toBe(revision + 1)
    expect(context.resolutionScale).toBe(
      webgpuQualityPresets.low.resolutionScale
    )
    expect(context.shapeDetail).toBe(false)
    expect(context.turbulence).toBe(false)
    expect(context.haze).toBe(true)
    expect(context.lightShafts).toBe(false)
    expect(context.clouds.maxIterationCount).toBe(
      webgpuQualityPresets.low.clouds.maxIterationCount
    )
    expect(context.maxIterationCountNode.value).toBe(
      webgpuQualityPresets.low.clouds.maxIterationCount
    )
    expect(context.maxIterationCountToSunNode.value).toBe(
      webgpuQualityPresets.low.clouds.maxIterationCountToSun
    )
    expect(context.minSecondaryStepSizeNode.value).toBe(
      webgpuQualityPresets.low.clouds.minSecondaryStepSize
    )
    expect(context.maxShadowFilterRadiusNode.value).toBe(
      webgpuQualityPresets.low.clouds.maxShadowFilterRadius
    )
    expect(context.maxShadowLengthIterationCountNode.value).toBe(
      webgpuQualityPresets.low.clouds.maxShadowLengthIterationCount
    )
    expect(context.minShadowLengthStepSizeNode.value).toBe(
      webgpuQualityPresets.low.clouds.minShadowLengthStepSize
    )
    expect(context.maxShadowLengthRayDistanceNode.value).toBe(
      webgpuQualityPresets.low.clouds.maxShadowLengthRayDistance
    )
    expect(context.hazeDensityScaleNode.value).toBe(
      webgpuQualityPresets.low.clouds.hazeDensityScale
    )
    expect(context.shadow.cascadeCount).toBe(
      webgpuQualityPresets.low.shadow.cascadeCount
    )
    expect(context.shadow.mapSize.toArray()).toEqual([
      webgpuQualityPresets.low.shadow.mapSize,
      webgpuQualityPresets.low.shadow.mapSize
    ])
    expect(context.shadow.maxIterationCount).toBe(
      webgpuQualityPresets.low.shadow.maxIterationCount
    )
  })

  test('keeps WebGPU quality presets within the shader loop budgets', () => {
    for (const preset of Object.values(webgpuQualityPresets)) {
      expect(preset.clouds.maxIterationCount).toBeLessThanOrEqual(
        WEBGPU_MAX_PRIMARY_STEPS
      )
      expect(preset.clouds.maxIterationCountToSun).toBeLessThanOrEqual(
        WEBGPU_MAX_SECONDARY_STEPS
      )
      expect(preset.clouds.maxIterationCountToGround).toBeLessThanOrEqual(
        WEBGPU_MAX_SECONDARY_STEPS
      )
    }
  })

  test('keeps ultra aligned with high cloud coverage semantics', () => {
    const { high, ultra } = webgpuQualityPresets

    expect(ultra.resolutionScale).toBe(high.resolutionScale)
    expect(ultra.shapeDetail).toBe(high.shapeDetail)
    expect(ultra.turbulence).toBe(high.turbulence)
    expect(ultra.haze).toBe(high.haze)
    expect(ultra.lightShafts).toBe(high.lightShafts)

    expect(ultra.clouds.maxRayDistance).toBe(high.clouds.maxRayDistance)
    expect(ultra.clouds.maxStepSize).toBe(high.clouds.maxStepSize)
    expect(ultra.clouds.perspectiveStepScale).toBe(
      high.clouds.perspectiveStepScale
    )
    expect(ultra.clouds.minDensity).toBe(high.clouds.minDensity)
    expect(ultra.clouds.minExtinction).toBe(high.clouds.minExtinction)
    expect(ultra.clouds.minTransmittance).toBe(high.clouds.minTransmittance)
    expect(ultra.clouds.maxIterationCountToGround).toBe(
      high.clouds.maxIterationCountToGround
    )
    expect(ultra.clouds.maxIterationCountToSun).toBe(
      high.clouds.maxIterationCountToSun
    )
    expect(ultra.clouds.minSecondaryStepSize).toBe(
      high.clouds.minSecondaryStepSize
    )
    expect(ultra.clouds.secondaryStepScale).toBe(
      high.clouds.secondaryStepScale
    )
    expect(ultra.clouds.maxShadowFilterRadius).toBe(
      high.clouds.maxShadowFilterRadius
    )
    expect(ultra.clouds.maxShadowLengthRayDistance).toBe(
      high.clouds.maxShadowLengthRayDistance
    )

    expect(ultra.clouds.maxIterationCount).toBeGreaterThanOrEqual(
      high.clouds.maxIterationCount
    )
    expect(ultra.clouds.minStepSize).toBeLessThan(high.clouds.minStepSize)
    expect(ultra.shadow.cascadeCount).toBe(high.shadow.cascadeCount)
    expect(ultra.shadow.maxIterationCount).toBe(high.shadow.maxIterationCount)
    expect(ultra.shadow.minDensity).toBe(high.shadow.minDensity)
    expect(ultra.shadow.mapSize).toBeGreaterThan(high.shadow.mapSize)
  })

  test('invalidates temporal history on texture and resolution changes', () => {
    const context = new CloudsContext()
    const texture = new Texture()
    const shapeTexture = new Data3DTexture(new Uint8Array([255]), 1, 1, 1)
    let revision = context.historyInvalidationRevision

    context.localWeatherTexture = texture
    expect(context.localWeatherTexture).toBe(texture)
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    revision = context.historyInvalidationRevision

    context.localWeatherTexture = texture
    expect(context.localWeatherTexture).toBe(texture)
    expect(context.historyInvalidationRevision).toBe(revision)

    context.shapeTexture = shapeTexture
    expect(context.shapeTexture).toBe(shapeTexture)
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    revision = context.historyInvalidationRevision

    context.stbnTexture = shapeTexture
    expect(context.stbnTexture).toBe(shapeTexture)
    expect(context.useStbnNode.value).toBe(true)
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    revision = context.historyInvalidationRevision

    context.resolutionScale = 0.5
    expect(context.resolutionScale).toBe(0.5)
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    revision = context.historyInvalidationRevision

    context.temporalUpscaleScale = 0.5
    expect(context.temporalUpscaleScale).toBe(0.5)
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    revision = context.historyInvalidationRevision

    context.temporalUpscaleScale = 0.5
    expect(context.historyInvalidationRevision).toBe(revision)
  })

  test('invalidates history for temporal and feature toggles only when values change', () => {
    const context = new CloudsContext()
    let revision = context.historyInvalidationRevision

    context.temporalUpscale = true
    expect(context.temporalUpscale).toBe(true)
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    revision = context.historyInvalidationRevision

    context.temporalUpscale = true
    expect(context.historyInvalidationRevision).toBe(revision)

    context.temporalAntialias = false
    expect(context.temporalAntialias).toBe(false)
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    revision = context.historyInvalidationRevision

    context.temporalAntialias = false
    expect(context.historyInvalidationRevision).toBe(revision)

    context.shapeDetail = false
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    revision = context.historyInvalidationRevision

    context.lightShafts = false
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    revision = context.historyInvalidationRevision

    context.turbulence = false
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    revision = context.historyInvalidationRevision

    context.haze = false
    expect(context.historyInvalidationRevision).toBe(revision + 1)
  })

  test('supports manual history invalidation without disturbing animated updates', () => {
    const context = new CloudsContext()
    const revision = context.historyInvalidationRevision

    context.localWeatherOffset.set(3, -2)
    context.shapeOffset.set(4, 5, 6)
    context.shapeDetailOffset.set(-1, -2, -3)

    context.invalidateHistory()
    expect(context.historyInvalidationRevision).toBe(revision + 1)
    expect(context.previousLocalWeatherOffset.toArray()).toEqual([3, -2])
    expect(context.previousShapeOffset.toArray()).toEqual([4, 5, 6])
    expect(context.previousShapeDetailOffset.toArray()).toEqual([-1, -2, -3])

    const updatedRevision = context.historyInvalidationRevision

    context.localWeatherVelocity.copy(new Vector2(2, -1))
    context.shapeVelocity.copy(new Vector3(1, 2, 3))
    context.shapeDetailVelocity.copy(new Vector3(-1, -2, -3))

    context.update(2)

    expect(context.historyInvalidationRevision).toBe(updatedRevision)
  })

  test('updates animated offsets and syncs toggle nodes', () => {
    const context = new CloudsContext()

    context.localWeatherVelocity.copy(new Vector2(2, -1))
    context.shapeVelocity.copy(new Vector3(1, 2, 3))
    context.shapeDetailVelocity.copy(new Vector3(-1, -2, -3))
    context.shapeDetail = false
    context.turbulence = false
    context.haze = false

    context.update(2)

    expect(context.localWeatherOffset.toArray()).toEqual([4, -2])
    expect(context.shapeOffset.toArray()).toEqual([2, 4, 6])
    expect(context.shapeDetailOffset.toArray()).toEqual([-2, -4, -6])
    expect(context.previousLocalWeatherOffset.toArray()).toEqual([0, 0])
    expect(context.previousShapeOffset.toArray()).toEqual([0, 0, 0])
    expect(context.previousShapeDetailOffset.toArray()).toEqual([0, 0, 0])
    expect(context.shapeDetailNode.value).toBe(false)
    expect(context.turbulenceNode.value).toBe(false)
    expect(context.hazeNode.value).toBe(false)
  })

  test('advances animated offsets only once per frame id', () => {
    const context = new CloudsContext()
    context.localWeatherVelocity.copy(new Vector2(2, 0))

    context.advance(1, 1)
    context.advance(1, 1)
    context.advance(2, 1)

    expect(context.localWeatherOffset.toArray()).toEqual([4, 0])
  })

  test('retrieves context from builder', () => {
    const context = new CloudsContext()
    const builder = {
      context: {
        getClouds: () => context
      }
    } as NodeBuilder

    expect(getCloudsContext(builder)).toBe(context)
  })

  test('throws when builder context is missing or invalid', () => {
    const missingBuilder = { context: {} } as NodeBuilder
    const invalidBuilder = {
      context: {
        getClouds: () => ({})
      }
    } as NodeBuilder

    expect(() => getCloudsContext(missingBuilder)).toThrow(
      'getClouds() was not found in the builder context.'
    )
    expect(() => getCloudsContext(invalidBuilder)).toThrow(
      'getClouds() must return an instanceof CloudsContext.'
    )
  })
})
