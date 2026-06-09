import { CloudLayers } from '@takram/three-clouds'

export const FUJI_PARITY_ANCHOR_LONGITUDE = 138.7274
export const FUJI_PARITY_ANCHOR_LATITUDE = 35.3606

export const FUJI_PARITY_POINT_OF_VIEW = {
  longitude: 138.634,
  latitude: 35.5,
  heading: -91,
  pitch: -27,
  distance: 8444
} as const

export const FUJI_NO_TILES_LOCAL_WEATHER_REPEAT = 24
export const FUJI_NO_TILES_LOCAL_WEATHER_OFFSETS = [
  [0, 0],
  [0.2, -0.1],
  [-0.18, 0.16],
  [0.35, 0.24]
] as const
export const FUJI_NO_TILES_LOCAL_WEATHER_OFFSET_INDEX = 2

export const FUJI_NO_TILES_LAYER_OPTIONS = [
  {
    altitude: 1_800,
    height: 900,
    densityScale: 0.22,
    shapeDetailAmount: 0.85,
    coverageFilterWidth: 0.68
  },
  {
    altitude: 2_600,
    height: 1_000,
    densityScale: 0.26,
    shapeAmount: 0.95,
    shapeDetailAmount: 0.7,
    coverageFilterWidth: 0.64
  },
  {
    altitude: 7_500,
    height: 700,
    densityScale: 0.008,
    shapeAmount: 0.55,
    shapeDetailAmount: 0,
    coverageFilterWidth: 0.5
  }
] as const

export const FUJI_NO_TILES_EFFECT_OPTIONS = {
  shapeRepeat: 0.0003,
  shapeDetailRepeat: 0.0045,
  turbulenceRepeat: 14,
  turbulenceDisplacement: 260,
  scatteringCoefficient: 1,
  absorptionCoefficient: 0,
  skyLightScale: 0.95,
  groundBounceScale: 0.6
} as const

interface FujiNoTilesCloudParameterTarget {
  localWeatherRepeat: { setScalar: (value: number) => unknown }
  localWeatherOffset: {
    x: number
    y: number
    set: (x: number, y: number) => unknown
  }
  previousLocalWeatherOffset?: {
    copy: (value: { x: number, y: number }) => unknown
  }
  shapeRepeat: { setScalar: (value: number) => unknown }
  shapeDetailRepeat: { setScalar: (value: number) => unknown }
  turbulenceRepeat: { setScalar: (value: number) => unknown }
  turbulenceDisplacement: number
  scatteringCoefficient: number
  absorptionCoefficient: number
  skyLightScale: number
  groundBounceScale: number
}

export function applyFujiNoTilesCloudLayerPreset(
  target: CloudLayers
): void {
  target.copy(CloudLayers.DEFAULT)
  target[0].set(FUJI_NO_TILES_LAYER_OPTIONS[0])
  target[1].set(FUJI_NO_TILES_LAYER_OPTIONS[1])
  target[2].set(FUJI_NO_TILES_LAYER_OPTIONS[2])
}

export function applyFujiNoTilesCloudParameterPreset(
  target: FujiNoTilesCloudParameterTarget
): void {
  const [weatherOffsetX, weatherOffsetY] =
    FUJI_NO_TILES_LOCAL_WEATHER_OFFSETS[
      FUJI_NO_TILES_LOCAL_WEATHER_OFFSET_INDEX
    ]
  target.localWeatherRepeat.setScalar(FUJI_NO_TILES_LOCAL_WEATHER_REPEAT)
  target.localWeatherOffset.set(weatherOffsetX, weatherOffsetY)
  target.previousLocalWeatherOffset?.copy(target.localWeatherOffset)
  target.shapeRepeat.setScalar(FUJI_NO_TILES_EFFECT_OPTIONS.shapeRepeat)
  target.shapeDetailRepeat.setScalar(
    FUJI_NO_TILES_EFFECT_OPTIONS.shapeDetailRepeat
  )
  target.turbulenceRepeat.setScalar(
    FUJI_NO_TILES_EFFECT_OPTIONS.turbulenceRepeat
  )
  target.turbulenceDisplacement =
    FUJI_NO_TILES_EFFECT_OPTIONS.turbulenceDisplacement
  target.scatteringCoefficient =
    FUJI_NO_TILES_EFFECT_OPTIONS.scatteringCoefficient
  target.absorptionCoefficient =
    FUJI_NO_TILES_EFFECT_OPTIONS.absorptionCoefficient
  target.skyLightScale = FUJI_NO_TILES_EFFECT_OPTIONS.skyLightScale
  target.groundBounceScale = FUJI_NO_TILES_EFFECT_OPTIONS.groundBounceScale
}

/**
 * Applies the shared Fuji no-tiles cloud preset to a mutable clouds target.
 *
 * @param target - WebGL `CloudsEffect` or WebGPU `CloudsContext`-like object
 *   that exposes the shared cloud parameter surface.
 * @returns Nothing. The target object is mutated in place.
 */
export function applyFujiNoTilesCloudPreset(target: {
  cloudLayers: CloudLayers
} & FujiNoTilesCloudParameterTarget): void {
  applyFujiNoTilesCloudLayerPreset(target.cloudLayers)
  applyFujiNoTilesCloudParameterPreset(target)
}
