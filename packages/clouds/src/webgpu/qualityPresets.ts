import type { QualityPreset } from '../qualityPresets'
import type { FrustumSplitMode } from '../helpers/splitFrustum'

export const WEBGPU_MAX_PRIMARY_STEPS = 256
export const WEBGPU_MAX_SECONDARY_STEPS = 8

export interface WebGPUCloudsQualityPreset {
  maxIterationCount: number
  minStepSize: number
  maxStepSize: number
  maxRayDistance: number
  maxShadowFilterRadius: number
  maxShadowLengthIterationCount: number
  minShadowLengthStepSize: number
  maxShadowLengthRayDistance: number
  perspectiveStepScale: number
  minDensity: number
  minExtinction: number
  minTransmittance: number
  hazeDensityScale: number
  hazeExponent: number
  hazeScatteringCoefficient: number
  hazeAbsorptionCoefficient: number
  maxIterationCountToGround: number
  maxIterationCountToSun: number
  minSecondaryStepSize: number
  secondaryStepScale: number
}

export interface WebGPUCloudShadowQualityPreset {
  cascadeCount: number
  mapSize: number
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
}

interface WebGPUQualityPresetDefinition {
  resolutionScale: number
  lightShafts: boolean
  shapeDetail: boolean
  turbulence: boolean
  haze: boolean
  clouds: WebGPUCloudsQualityPreset
  shadow: WebGPUCloudShadowQualityPreset
}

const highClouds: WebGPUCloudsQualityPreset = {
  maxIterationCount: 60,
  minStepSize: 30,
  maxStepSize: 900,
  maxRayDistance: 2e5,
  maxShadowFilterRadius: 6,
  maxShadowLengthIterationCount: WEBGPU_MAX_PRIMARY_STEPS,
  minShadowLengthStepSize: 80,
  maxShadowLengthRayDistance: 2e5,
  perspectiveStepScale: 1.001,
  minDensity: 1e-5,
  minExtinction: 1e-5,
  minTransmittance: 1e-2,
  hazeDensityScale: 3e-5,
  hazeExponent: 1e-3,
  hazeScatteringCoefficient: 0.9,
  hazeAbsorptionCoefficient: 0.5,
  maxIterationCountToGround: 2,
  maxIterationCountToSun: 3,
  minSecondaryStepSize: 80,
  secondaryStepScale: 1.8
}

const highShadow: WebGPUCloudShadowQualityPreset = {
  cascadeCount: 3,
  mapSize: 512,
  maxFar: null,
  farScale: 1,
  splitMode: 'practical',
  splitLambda: 0.6,
  margin: 0,
  fade: true,
  maxIterationCount: 50,
  minStepSize: 100,
  maxStepSize: 1000,
  minDensity: 1e-5,
  minExtinction: 1e-5,
  minTransmittance: 1e-4,
  opticalDepthTailScale: 2
}

export const webgpuQualityPresets: Record<
  QualityPreset,
  WebGPUQualityPresetDefinition
> = {
  low: {
    resolutionScale: 1,
    lightShafts: false,
    shapeDetail: false,
    turbulence: false,
    haze: true,
    clouds: {
      ...highClouds,
      maxIterationCount: 32,
      minStepSize: 84,
      maxStepSize: 920,
      maxRayDistance: 1.5e5,
      maxShadowFilterRadius: 4,
      maxShadowLengthIterationCount: 24,
      minShadowLengthStepSize: 160,
      maxShadowLengthRayDistance: 1.5e5,
      perspectiveStepScale: 1.006,
      minDensity: 1e-4,
      minExtinction: 1e-4,
      minTransmittance: 8e-2,
      maxIterationCountToGround: 0,
      maxIterationCountToSun: 1,
      minSecondaryStepSize: 140,
      secondaryStepScale: 2.1
    },
    shadow: {
      ...highShadow,
      cascadeCount: 2,
      mapSize: 256,
      maxIterationCount: 25,
      minDensity: 1e-4,
      minExtinction: 1e-4,
      minTransmittance: 1e-2
    }
  },
  medium: {
    resolutionScale: 1,
    lightShafts: false,
    shapeDetail: true,
    turbulence: false,
    haze: true,
    clouds: {
      ...highClouds,
      maxIterationCount: 46,
      minStepSize: 54,
      maxStepSize: 760,
      maxShadowFilterRadius: 5,
      maxShadowLengthIterationCount: 40,
      minShadowLengthStepSize: 120,
      perspectiveStepScale: 1.003,
      minDensity: 1e-4,
      minExtinction: 1e-4,
      minTransmittance: 3e-2,
      maxIterationCountToGround: 1,
      maxIterationCountToSun: 2,
      minSecondaryStepSize: 104,
      secondaryStepScale: 1.95
    },
    shadow: {
      ...highShadow,
      mapSize: 256,
      minDensity: 1e-4,
      minExtinction: 1e-4
    }
  },
  high: {
    resolutionScale: 1,
    lightShafts: true,
    shapeDetail: true,
    turbulence: true,
    haze: true,
    clouds: highClouds,
    shadow: highShadow
  },
  ultra: {
    resolutionScale: 1,
    lightShafts: true,
    shapeDetail: true,
    turbulence: true,
    haze: true,
    clouds: {
      ...highClouds,
      maxIterationCount: WEBGPU_MAX_PRIMARY_STEPS,
      minStepSize: 24,
      maxShadowLengthIterationCount: WEBGPU_MAX_PRIMARY_STEPS,
      minShadowLengthStepSize: 64
    },
    shadow: {
      ...highShadow,
      mapSize: 1024
    }
  }
}
