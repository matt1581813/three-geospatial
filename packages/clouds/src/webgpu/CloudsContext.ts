import {
  Data3DTexture,
  DataTexture,
  LinearFilter,
  NearestFilter,
  NoColorSpace,
  RedFormat,
  RepeatWrapping,
  RGBAFormat,
  Vector2,
  Vector3,
  Vector4,
  type Texture
} from 'three'
import { uniform } from 'three/tsl'
import type { NodeBuilder } from 'three/webgpu'

import { CloudLayers } from '../CloudLayers'
import type { FrustumSplitMode } from '../helpers/splitFrustum'
import type { QualityPreset } from '../qualityPresets'
import { createCloudLayerUniforms, updateCloudLayerUniforms } from '../uniforms'
import { webgpuQualityPresets } from './qualityPresets'

export const fallbackLocalWeatherTexture = /*#__PURE__*/ create2DTexture([
  255, 255, 255, 255
])
export const fallbackTurbulenceTexture = /*#__PURE__*/ create2DTexture([
  128, 128, 128, 255
])
export const fallbackShapeTexture = /*#__PURE__*/ create3DTexture([255])
export const fallbackShapeDetailTexture = /*#__PURE__*/ create3DTexture([255])
export const fallbackStbnTexture = /*#__PURE__*/ createStbnTexture()

function create2DTexture(data: ArrayLike<number>): DataTexture {
  const texture = new DataTexture(Uint8Array.from(data), 1, 1, RGBAFormat)
  texture.minFilter = LinearFilter
  texture.magFilter = LinearFilter
  texture.wrapS = RepeatWrapping
  texture.wrapT = RepeatWrapping
  texture.colorSpace = NoColorSpace
  texture.needsUpdate = true
  return texture
}

function create3DTexture(data: ArrayLike<number>): Data3DTexture {
  const texture = new Data3DTexture(Uint8Array.from(data), 1, 1, 1)
  texture.format = RedFormat
  texture.minFilter = LinearFilter
  texture.magFilter = LinearFilter
  texture.wrapS = RepeatWrapping
  texture.wrapT = RepeatWrapping
  texture.wrapR = RepeatWrapping
  texture.colorSpace = NoColorSpace
  texture.needsUpdate = true
  return texture
}

function createStbnTexture(width = 32, height = 32, depth = 16): Data3DTexture {
  const data = new Uint8Array(width * height * depth)
  let seed = 0x12345678
  const random = (): number => {
    seed ^= seed << 13
    seed ^= seed >>> 17
    seed ^= seed << 5
    return (seed >>> 0) / 0xffffffff
  }

  for (let z = 0; z < depth; ++z) {
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        const index = z * width * height + y * width + x
        data[index] = Math.round(255 * random())
      }
    }
  }

  const texture = new Data3DTexture(data, width, height, depth)
  texture.format = RedFormat
  texture.minFilter = NearestFilter
  texture.magFilter = NearestFilter
  texture.wrapS = RepeatWrapping
  texture.wrapT = RepeatWrapping
  texture.wrapR = RepeatWrapping
  texture.unpackAlignment = 1
  texture.colorSpace = NoColorSpace
  texture.needsUpdate = true
  return texture
}

function setChannelMask(channel: string, target: Vector4): void {
  target.set(
    channel === 'r' ? 1 : 0,
    channel === 'g' ? 1 : 0,
    channel === 'b' ? 1 : 0,
    channel === 'a' ? 1 : 0
  )
}

export class CloudsContext {
  readonly cloudLayers = CloudLayers.DEFAULT.clone()

  correctAltitude = true

  readonly localWeatherRepeat = new Vector2().setScalar(100)
  readonly localWeatherOffset = new Vector2()
  readonly previousLocalWeatherOffset = new Vector2()
  readonly shapeRepeat = new Vector3().setScalar(0.0003)
  readonly shapeOffset = new Vector3()
  readonly previousShapeOffset = new Vector3()
  readonly shapeDetailRepeat = new Vector3().setScalar(0.006)
  readonly shapeDetailOffset = new Vector3()
  readonly previousShapeDetailOffset = new Vector3()
  readonly turbulenceRepeat = new Vector2().setScalar(20)

  readonly localWeatherVelocity = new Vector2()
  readonly shapeVelocity = new Vector3()
  readonly shapeDetailVelocity = new Vector3()

  scatteringCoefficient = 1
  absorptionCoefficient = 0
  coverage = 0.3
  turbulenceDisplacement = 350
  animateStbn = true
  stbnFrameIndex = 0

  skyLightScale = 1
  groundBounceScale = 1
  powderScale = 0.8
  powderExponent = 150
  scatterAnisotropy1 = 0.7
  scatterAnisotropy2 = -0.2
  scatterAnisotropyMix = 0.5

  readonly clouds = {
    ...webgpuQualityPresets.high.clouds
  }
  readonly shadow: {
    cascadeCount: number
    mapSize: Vector2
    maxFar: number | null
    farScale: number
    splitMode: FrustumSplitMode
    splitLambda: number
    margin: number
    fade: boolean
    maxIterationCount: number
    minStepSize: number
    maxStepSize: number
    minDensity: number
    minExtinction: number
    minTransmittance: number
    opticalDepthTailScale: number
  } = {
    ...webgpuQualityPresets.high.shadow,
    mapSize: new Vector2().setScalar(webgpuQualityPresets.high.shadow.mapSize)
  }

  private _temporalUpscale = false
  private _temporalAntialias = true
  private _qualityPreset: QualityPreset = 'high'
  private _resolutionScale = webgpuQualityPresets.high.resolutionScale
  private _temporalUpscaleScale = 0.375
  private _discardAllHistory = false
  private _velocityThresholdPixels = 6
  private _historyResetDistanceThreshold = 100
  private _temporalAlpha: number | null = null
  private _lightShafts = webgpuQualityPresets.high.lightShafts
  private _shapeDetail = webgpuQualityPresets.high.shapeDetail
  private _turbulence = webgpuQualityPresets.high.turbulence
  private _haze = webgpuQualityPresets.high.haze
  private _historyInvalidationRevision = 0
  private _lastAdvancedFrameId = -1

  private _localWeatherTexture: Texture | null = null
  private _shapeTexture: Data3DTexture | null = null
  private _shapeDetailTexture: Data3DTexture | null = null
  private _turbulenceTexture: Texture | null = null
  private _stbnTexture: Data3DTexture | null = null

  private readonly layerState = createCloudLayerUniforms()

  readonly correctAltitudeNode = uniform(this.correctAltitude).setName(
    'cloudsCorrectAltitude'
  )
  readonly scatteringCoefficientNode = uniform(
    this.scatteringCoefficient
  ).setName('cloudsScatteringCoefficient')
  readonly absorptionCoefficientNode = uniform(
    this.absorptionCoefficient
  ).setName('cloudsAbsorptionCoefficient')
  readonly coverageNode = uniform(this.coverage).setName('cloudsCoverage')
  readonly localWeatherRepeatNode = uniform(this.localWeatherRepeat).setName(
    'cloudsLocalWeatherRepeat'
  )
  readonly localWeatherOffsetNode = uniform(this.localWeatherOffset).setName(
    'cloudsLocalWeatherOffset'
  )
  readonly previousLocalWeatherOffsetNode = uniform(
    this.previousLocalWeatherOffset
  ).setName('cloudsPreviousLocalWeatherOffset')
  readonly shapeRepeatNode = uniform(this.shapeRepeat).setName(
    'cloudsShapeRepeat'
  )
  readonly shapeOffsetNode = uniform(this.shapeOffset).setName(
    'cloudsShapeOffset'
  )
  readonly previousShapeOffsetNode = uniform(this.previousShapeOffset).setName(
    'cloudsPreviousShapeOffset'
  )
  readonly shapeDetailRepeatNode = uniform(this.shapeDetailRepeat).setName(
    'cloudsShapeDetailRepeat'
  )
  readonly shapeDetailOffsetNode = uniform(this.shapeDetailOffset).setName(
    'cloudsShapeDetailOffset'
  )
  readonly previousShapeDetailOffsetNode = uniform(
    this.previousShapeDetailOffset
  ).setName('cloudsPreviousShapeDetailOffset')
  readonly turbulenceRepeatNode = uniform(this.turbulenceRepeat).setName(
    'cloudsTurbulenceRepeat'
  )
  readonly turbulenceDisplacementNode = uniform(
    this.turbulenceDisplacement
  ).setName('cloudsTurbulenceDisplacement')

  readonly shapeDetailNode = uniform(this.shapeDetail).setName(
    'cloudsShapeDetailEnabled'
  )
  readonly turbulenceNode = uniform(this.turbulence).setName(
    'cloudsTurbulenceEnabled'
  )
  readonly hazeNode = uniform(this.haze).setName('cloudsHazeEnabled')
  readonly lightShaftsNode = uniform(this.lightShafts).setName(
    'cloudsLightShaftsEnabled'
  )
  readonly useStbnNode = uniform(false).setName('cloudsUseStbn')

  readonly skyLightScaleNode = uniform(this.skyLightScale).setName(
    'cloudsSkyLightScale'
  )
  readonly groundBounceScaleNode = uniform(this.groundBounceScale).setName(
    'cloudsGroundBounceScale'
  )
  readonly powderScaleNode = uniform(this.powderScale).setName(
    'cloudsPowderScale'
  )
  readonly powderExponentNode = uniform(this.powderExponent).setName(
    'cloudsPowderExponent'
  )
  readonly scatterAnisotropy1Node = uniform(this.scatterAnisotropy1).setName(
    'cloudsScatterAnisotropy1'
  )
  readonly scatterAnisotropy2Node = uniform(this.scatterAnisotropy2).setName(
    'cloudsScatterAnisotropy2'
  )
  readonly scatterAnisotropyMixNode = uniform(
    this.scatterAnisotropyMix
  ).setName('cloudsScatterAnisotropyMix')

  readonly maxIterationCountNode = uniform(
    this.clouds.maxIterationCount,
    'int'
  ).setName('cloudsMaxIterationCount')
  readonly minStepSizeNode = uniform(this.clouds.minStepSize).setName(
    'cloudsMinStepSize'
  )
  readonly maxStepSizeNode = uniform(this.clouds.maxStepSize).setName(
    'cloudsMaxStepSize'
  )
  readonly maxRayDistanceNode = uniform(this.clouds.maxRayDistance).setName(
    'cloudsMaxRayDistance'
  )
  readonly maxShadowFilterRadiusNode = uniform(
    this.clouds.maxShadowFilterRadius
  ).setName('cloudsMaxShadowFilterRadius')
  readonly maxShadowLengthIterationCountNode = uniform(
    this.clouds.maxShadowLengthIterationCount,
    'int'
  ).setName('cloudsMaxShadowLengthIterationCount')
  readonly minShadowLengthStepSizeNode = uniform(
    this.clouds.minShadowLengthStepSize
  ).setName('cloudsMinShadowLengthStepSize')
  readonly maxShadowLengthRayDistanceNode = uniform(
    this.clouds.maxShadowLengthRayDistance
  ).setName('cloudsMaxShadowLengthRayDistance')
  readonly maxIterationCountToSunNode = uniform(
    this.clouds.maxIterationCountToSun,
    'int'
  ).setName('cloudsMaxIterationCountToSun')
  readonly maxIterationCountToGroundNode = uniform(
    this.clouds.maxIterationCountToGround,
    'int'
  ).setName('cloudsMaxIterationCountToGround')
  readonly minSecondaryStepSizeNode = uniform(
    this.clouds.minSecondaryStepSize
  ).setName('cloudsMinSecondaryStepSize')
  readonly secondaryStepScaleNode = uniform(
    this.clouds.secondaryStepScale
  ).setName('cloudsSecondaryStepScale')
  readonly perspectiveStepScaleNode = uniform(
    this.clouds.perspectiveStepScale
  ).setName('cloudsPerspectiveStepScale')
  readonly minDensityNode = uniform(this.clouds.minDensity).setName(
    'cloudsMinDensity'
  )
  readonly minExtinctionNode = uniform(this.clouds.minExtinction).setName(
    'cloudsMinExtinction'
  )
  readonly minTransmittanceNode = uniform(this.clouds.minTransmittance).setName(
    'cloudsMinTransmittance'
  )
  readonly hazeDensityScaleNode = uniform(this.clouds.hazeDensityScale).setName(
    'cloudsHazeDensityScale'
  )
  readonly hazeExponentNode = uniform(this.clouds.hazeExponent).setName(
    'cloudsHazeExponent'
  )
  readonly hazeScatteringCoefficientNode = uniform(
    this.clouds.hazeScatteringCoefficient
  ).setName('cloudsHazeScatteringCoefficient')
  readonly hazeAbsorptionCoefficientNode = uniform(
    this.clouds.hazeAbsorptionCoefficient
  ).setName('cloudsHazeAbsorptionCoefficient')

  readonly minLayerHeightsNode = uniform(
    this.layerState.minLayerHeights.value
  ).setName('cloudsMinLayerHeights')
  readonly maxLayerHeightsNode = uniform(
    this.layerState.maxLayerHeights.value
  ).setName('cloudsMaxLayerHeights')
  readonly minIntervalHeightsNode = uniform(
    this.layerState.minIntervalHeights.value
  ).setName('cloudsMinIntervalHeights')
  readonly maxIntervalHeightsNode = uniform(
    this.layerState.maxIntervalHeights.value
  ).setName('cloudsMaxIntervalHeights')
  readonly densityScalesNode = uniform(
    this.layerState.densityScales.value
  ).setName('cloudsDensityScales')
  readonly shapeAmountsNode = uniform(
    this.layerState.shapeAmounts.value
  ).setName('cloudsShapeAmounts')
  readonly shapeDetailAmountsNode = uniform(
    this.layerState.shapeDetailAmounts.value
  ).setName('cloudsShapeDetailAmounts')
  readonly weatherExponentsNode = uniform(
    this.layerState.weatherExponents.value
  ).setName('cloudsWeatherExponents')
  readonly shapeAlteringBiasesNode = uniform(
    this.layerState.shapeAlteringBiases.value
  ).setName('cloudsShapeAlteringBiases')
  readonly coverageFilterWidthsNode = uniform(
    this.layerState.coverageFilterWidths.value
  ).setName('cloudsCoverageFilterWidths')
  readonly minHeightNode = uniform(this.layerState.minHeight.value).setName(
    'cloudsMinHeight'
  )
  readonly maxHeightNode = uniform(this.layerState.maxHeight.value).setName(
    'cloudsMaxHeight'
  )
  readonly shadowMinHeightNode = uniform(0).setName('cloudsShadowMinHeight')
  readonly shadowMaxHeightNode = uniform(0).setName('cloudsShadowMaxHeight')
  readonly densityProfileExpTermsNode = uniform(
    this.layerState.densityProfile.value.expTerms
  ).setName('cloudsDensityProfileExpTerms')
  readonly densityProfileExponentsNode = uniform(
    this.layerState.densityProfile.value.exponents
  ).setName('cloudsDensityProfileExponents')
  readonly densityProfileLinearTermsNode = uniform(
    this.layerState.densityProfile.value.linearTerms
  ).setName('cloudsDensityProfileLinearTerms')
  readonly densityProfileConstantTermsNode = uniform(
    this.layerState.densityProfile.value.constantTerms
  ).setName('cloudsDensityProfileConstantTerms')

  readonly localWeatherChannelMask0Node = uniform(new Vector4()).setName(
    'cloudsLocalWeatherChannelMask0'
  )
  readonly localWeatherChannelMask1Node = uniform(new Vector4()).setName(
    'cloudsLocalWeatherChannelMask1'
  )
  readonly localWeatherChannelMask2Node = uniform(new Vector4()).setName(
    'cloudsLocalWeatherChannelMask2'
  )
  readonly localWeatherChannelMask3Node = uniform(new Vector4()).setName(
    'cloudsLocalWeatherChannelMask3'
  )
  readonly shadowLayerMaskNode = uniform(new Vector4()).setName(
    'cloudsShadowLayerMask'
  )

  constructor() {
    this.syncNodes()
    this.invalidateHistory()
  }

  get temporalUpscale(): boolean {
    return this._temporalUpscale
  }

  set temporalUpscale(value: boolean) {
    if (this._temporalUpscale === value) {
      return
    }
    this._temporalUpscale = value
    this.invalidateHistory()
  }

  get temporalAntialias(): boolean {
    return this._temporalAntialias
  }

  set temporalAntialias(value: boolean) {
    if (this._temporalAntialias === value) {
      return
    }
    this._temporalAntialias = value
    this.invalidateHistory()
  }

  get historyInvalidationRevision(): number {
    return this._historyInvalidationRevision
  }

  get qualityPreset(): QualityPreset {
    return this._qualityPreset
  }

  set qualityPreset(value: QualityPreset) {
    if (this._qualityPreset === value) {
      return
    }
    this._qualityPreset = value
    const preset = webgpuQualityPresets[value]
    this._resolutionScale = preset.resolutionScale
    this._lightShafts = preset.lightShafts
    this._shapeDetail = preset.shapeDetail
    this._turbulence = preset.turbulence
    this._haze = preset.haze
    Object.assign(this.clouds, preset.clouds)
    this.shadow.cascadeCount = preset.shadow.cascadeCount
    this.shadow.mapSize.setScalar(preset.shadow.mapSize)
    this.shadow.maxFar = preset.shadow.maxFar
    this.shadow.farScale = preset.shadow.farScale
    this.shadow.splitMode = preset.shadow.splitMode
    this.shadow.splitLambda = preset.shadow.splitLambda
    this.shadow.margin = preset.shadow.margin
    this.shadow.fade = preset.shadow.fade
    this.shadow.maxIterationCount = preset.shadow.maxIterationCount
    this.shadow.minStepSize = preset.shadow.minStepSize
    this.shadow.maxStepSize = preset.shadow.maxStepSize
    this.shadow.minDensity = preset.shadow.minDensity
    this.shadow.minExtinction = preset.shadow.minExtinction
    this.shadow.minTransmittance = preset.shadow.minTransmittance
    this.shadow.opticalDepthTailScale = preset.shadow.opticalDepthTailScale
    this.syncNodes()
    this.invalidateHistory()
  }

  get resolutionScale(): number {
    return this._resolutionScale
  }

  set resolutionScale(value: number) {
    const next = Math.max(value, 1 / 16)
    if (this._resolutionScale === next) {
      return
    }
    this._resolutionScale = next
    this.invalidateHistory()
  }

  get temporalUpscaleScale(): number {
    return this._temporalUpscaleScale
  }

  set temporalUpscaleScale(value: number) {
    const next = Math.min(Math.max(value, 1 / 16), 1)
    if (this._temporalUpscaleScale === next) {
      return
    }
    this._temporalUpscaleScale = next
    this.invalidateHistory()
  }

  get discardAllHistory(): boolean {
    return this._discardAllHistory
  }

  set discardAllHistory(value: boolean) {
    if (this._discardAllHistory === value) {
      return
    }
    this._discardAllHistory = value
    this.invalidateHistory()
  }

  get velocityThresholdPixels(): number {
    return this._velocityThresholdPixels
  }

  set velocityThresholdPixels(value: number) {
    const next = Math.max(value, 0)
    if (this._velocityThresholdPixels === next) {
      return
    }
    this._velocityThresholdPixels = next
    this.invalidateHistory()
  }

  get historyResetDistanceThreshold(): number {
    return this._historyResetDistanceThreshold
  }

  set historyResetDistanceThreshold(value: number) {
    const next = Math.max(value, 0)
    if (this._historyResetDistanceThreshold === next) {
      return
    }
    this._historyResetDistanceThreshold = next
    this.invalidateHistory()
  }

  get temporalAlpha(): number | null {
    return this._temporalAlpha
  }

  set temporalAlpha(value: number | null) {
    const next = value == null ? null : Math.min(Math.max(value, 0), 1)
    if (this._temporalAlpha === next) {
      return
    }
    this._temporalAlpha = next
    this.invalidateHistory()
  }

  get shapeDetail(): boolean {
    return this._shapeDetail
  }

  get lightShafts(): boolean {
    return this._lightShafts
  }

  set lightShafts(value: boolean) {
    if (this._lightShafts === value) {
      return
    }
    this._lightShafts = value
    this.invalidateHistory()
  }

  set shapeDetail(value: boolean) {
    if (this._shapeDetail === value) {
      return
    }
    this._shapeDetail = value
    this.invalidateHistory()
  }

  get turbulence(): boolean {
    return this._turbulence
  }

  set turbulence(value: boolean) {
    if (this._turbulence === value) {
      return
    }
    this._turbulence = value
    this.invalidateHistory()
  }

  get haze(): boolean {
    return this._haze
  }

  set haze(value: boolean) {
    if (this._haze === value) {
      return
    }
    this._haze = value
    this.invalidateHistory()
  }

  get localWeatherTexture(): Texture | null {
    return this._localWeatherTexture
  }

  set localWeatherTexture(value: Texture | null) {
    if (this._localWeatherTexture === value) {
      return
    }
    this._localWeatherTexture = value
    this.invalidateHistory()
  }

  get shapeTexture(): Data3DTexture | null {
    return this._shapeTexture
  }

  set shapeTexture(value: Data3DTexture | null) {
    if (this._shapeTexture === value) {
      return
    }
    this._shapeTexture = value
    this.invalidateHistory()
  }

  get shapeDetailTexture(): Data3DTexture | null {
    return this._shapeDetailTexture
  }

  set shapeDetailTexture(value: Data3DTexture | null) {
    if (this._shapeDetailTexture === value) {
      return
    }
    this._shapeDetailTexture = value
    this.invalidateHistory()
  }

  get turbulenceTexture(): Texture | null {
    return this._turbulenceTexture
  }

  set turbulenceTexture(value: Texture | null) {
    if (this._turbulenceTexture === value) {
      return
    }
    this._turbulenceTexture = value
    this.invalidateHistory()
  }

  get stbnTexture(): Data3DTexture | null {
    return this._stbnTexture
  }

  set stbnTexture(value: Data3DTexture | null) {
    if (this._stbnTexture === value) {
      return
    }
    this._stbnTexture = value
    this.useStbnNode.value = value != null
    this.invalidateHistory()
  }

  get resolvedLocalWeatherTexture(): Texture {
    return this._localWeatherTexture ?? fallbackLocalWeatherTexture
  }

  get resolvedShapeTexture(): Data3DTexture {
    return this._shapeTexture ?? fallbackShapeTexture
  }

  get resolvedShapeDetailTexture(): Data3DTexture {
    return this._shapeDetailTexture ?? fallbackShapeDetailTexture
  }

  get resolvedTurbulenceTexture(): Texture {
    return this._turbulenceTexture ?? fallbackTurbulenceTexture
  }

  get resolvedStbnTexture(): Data3DTexture {
    return this._stbnTexture ?? fallbackStbnTexture
  }

  update(deltaTime = 0): void {
    this.previousLocalWeatherOffset.copy(this.localWeatherOffset)
    this.previousShapeOffset.copy(this.shapeOffset)
    this.previousShapeDetailOffset.copy(this.shapeDetailOffset)
    if (deltaTime > 0) {
      this.localWeatherOffset.addScaledVector(
        this.localWeatherVelocity,
        deltaTime
      )
      this.shapeOffset.addScaledVector(this.shapeVelocity, deltaTime)
      this.shapeDetailOffset.addScaledVector(
        this.shapeDetailVelocity,
        deltaTime
      )
    }
    this.syncNodes()
  }

  advance(frameId: number, deltaTime = 0): void {
    if (this._lastAdvancedFrameId === frameId) {
      this.syncNodes()
      return
    }
    this._lastAdvancedFrameId = frameId
    this.update(deltaTime)
  }

  dispose(): void {}

  invalidateHistory(): void {
    this.previousLocalWeatherOffset.copy(this.localWeatherOffset)
    this.previousShapeOffset.copy(this.shapeOffset)
    this.previousShapeDetailOffset.copy(this.shapeDetailOffset)
    this._historyInvalidationRevision += 1
  }

  private syncNodes(): void {
    this.correctAltitudeNode.value = this.correctAltitude
    this.scatteringCoefficientNode.value = this.scatteringCoefficient
    this.absorptionCoefficientNode.value = this.absorptionCoefficient
    this.coverageNode.value = this.coverage
    this.turbulenceDisplacementNode.value = this.turbulenceDisplacement

    this.shapeDetailNode.value = this.shapeDetail
    this.turbulenceNode.value = this.turbulence
    this.hazeNode.value = this.haze
    this.lightShaftsNode.value = this.lightShafts
    this.useStbnNode.value = this._stbnTexture != null

    this.skyLightScaleNode.value = this.skyLightScale
    this.groundBounceScaleNode.value = this.groundBounceScale
    this.powderScaleNode.value = this.powderScale
    this.powderExponentNode.value = this.powderExponent
    this.scatterAnisotropy1Node.value = this.scatterAnisotropy1
    this.scatterAnisotropy2Node.value = this.scatterAnisotropy2
    this.scatterAnisotropyMixNode.value = this.scatterAnisotropyMix

    this.maxIterationCountNode.value = Math.max(
      1,
      Math.round(this.clouds.maxIterationCount)
    )
    this.maxIterationCountToSunNode.value = Math.max(
      0,
      Math.round(this.clouds.maxIterationCountToSun)
    )
    this.maxIterationCountToGroundNode.value = Math.max(
      0,
      Math.round(this.clouds.maxIterationCountToGround)
    )
    this.minStepSizeNode.value = this.clouds.minStepSize
    this.maxStepSizeNode.value = this.clouds.maxStepSize
    this.maxRayDistanceNode.value = this.clouds.maxRayDistance
    this.maxShadowFilterRadiusNode.value = this.clouds.maxShadowFilterRadius
    this.maxShadowLengthIterationCountNode.value = Math.max(
      1,
      Math.round(this.clouds.maxShadowLengthIterationCount)
    )
    this.minShadowLengthStepSizeNode.value = this.clouds.minShadowLengthStepSize
    this.maxShadowLengthRayDistanceNode.value =
      this.clouds.maxShadowLengthRayDistance
    this.minSecondaryStepSizeNode.value = this.clouds.minSecondaryStepSize
    this.secondaryStepScaleNode.value = this.clouds.secondaryStepScale
    this.perspectiveStepScaleNode.value = this.clouds.perspectiveStepScale
    this.minDensityNode.value = this.clouds.minDensity
    this.minExtinctionNode.value = this.clouds.minExtinction
    this.minTransmittanceNode.value = this.clouds.minTransmittance
    this.hazeDensityScaleNode.value = this.clouds.hazeDensityScale
    this.hazeExponentNode.value = this.clouds.hazeExponent
    this.hazeScatteringCoefficientNode.value =
      this.clouds.hazeScatteringCoefficient
    this.hazeAbsorptionCoefficientNode.value =
      this.clouds.hazeAbsorptionCoefficient

    updateCloudLayerUniforms(this.layerState, this.cloudLayers)
    this.minHeightNode.value = this.layerState.minHeight.value
    this.maxHeightNode.value = this.layerState.maxHeight.value

    let shadowMinHeight = Number.POSITIVE_INFINITY
    let shadowMaxHeight = 0
    this.shadowLayerMaskNode.value.set(0, 0, 0, 0)
    for (let index = 0; index < this.cloudLayers.length; ++index) {
      const layer = this.cloudLayers[index]
      const enabled = layer.shadow && layer.height > 0
      this.shadowLayerMaskNode.value.setComponent(index, enabled ? 1 : 0)
      if (!enabled) {
        continue
      }
      shadowMinHeight = Math.min(shadowMinHeight, layer.altitude)
      shadowMaxHeight = Math.max(shadowMaxHeight, layer.altitude + layer.height)
    }
    if (shadowMinHeight === Number.POSITIVE_INFINITY) {
      this.shadowMinHeightNode.value = 0
      this.shadowMaxHeightNode.value = 0
    } else {
      this.shadowMinHeightNode.value = shadowMinHeight
      this.shadowMaxHeightNode.value = shadowMaxHeight
    }

    setChannelMask(
      this.cloudLayers[0].channel,
      this.localWeatherChannelMask0Node.value
    )
    setChannelMask(
      this.cloudLayers[1].channel,
      this.localWeatherChannelMask1Node.value
    )
    setChannelMask(
      this.cloudLayers[2].channel,
      this.localWeatherChannelMask2Node.value
    )
    setChannelMask(
      this.cloudLayers[3].channel,
      this.localWeatherChannelMask3Node.value
    )
  }
}

export function getCloudsContext(builder: NodeBuilder): CloudsContext {
  if (typeof builder.context.getClouds !== 'function') {
    throw new Error('getClouds() was not found in the builder context.')
  }
  const context = builder.context.getClouds()
  if (!(context instanceof CloudsContext)) {
    throw new Error('getClouds() must return an instanceof CloudsContext.')
  }
  return context
}
