import type { CloudsContext } from '@takram/three-clouds/webgpu'

export type CloudStoryViewPreset = 'ground' | 'cruise'

interface StoryLayerPreset {
  altitude: number
  height: number
  densityScale: number
  shapeAmount?: number
  shapeDetailAmount: number
}

interface StoryScenePreset {
  coverage: number
  scatteringCoefficient: number
  absorptionCoefficient: number
  skyLightScale: number
  groundBounceScale: number
  localWeatherRepeat: number
  shapeDetailRepeat: number
  turbulenceDisplacement: number
  layers: [StoryLayerPreset, StoryLayerPreset, StoryLayerPreset]
}

const scenePresets: Record<CloudStoryViewPreset, StoryScenePreset> = {
  ground: {
    coverage: 0.41,
    scatteringCoefficient: 0.95,
    absorptionCoefficient: 0.06,
    skyLightScale: 0.9,
    groundBounceScale: 0.18,
    localWeatherRepeat: 31,
    shapeDetailRepeat: 0.0038,
    turbulenceDisplacement: 220,
    layers: [
      {
        altitude: 3_000,
        height: 1_000,
        densityScale: 0.14,
        shapeDetailAmount: 0.5
      },
      {
        altitude: 4_600,
        height: 1_550,
        densityScale: 0.14,
        shapeDetailAmount: 0.42
      },
      {
        altitude: 8_500,
        height: 1_300,
        densityScale: 0.0038,
        shapeAmount: 0.5,
        shapeDetailAmount: 0
      }
    ]
  },
  cruise: {
    coverage: 0.47,
    scatteringCoefficient: 0.95,
    absorptionCoefficient: 0.06,
    skyLightScale: 0.9,
    groundBounceScale: 0.28,
    localWeatherRepeat: 28,
    shapeDetailRepeat: 0.0036,
    turbulenceDisplacement: 230,
    layers: [
      {
        altitude: 3_200,
        height: 950,
        densityScale: 0.055,
        shapeDetailAmount: 0.44
      },
      {
        altitude: 5_900,
        height: 1_850,
        densityScale: 0.16,
        shapeDetailAmount: 0.38
      },
      {
        altitude: 8_300,
        height: 1_850,
        densityScale: 0.012,
        shapeAmount: 0.54,
        shapeDetailAmount: 0
      }
    ]
  }
}

export function applyCloudStoryPreset(
  cloudsContext: CloudsContext,
  viewPreset: CloudStoryViewPreset
): void {
  const scene = scenePresets[viewPreset]
  let changed = false

  const assign = <T>(target: T, next: T, apply: (value: T) => void): void => {
    if (target === next) {
      return
    }
    apply(next)
    changed = true
  }

  scene.layers.forEach((layerPreset, index) => {
    const layer = cloudsContext.cloudLayers[index]
    assign(layer.altitude, layerPreset.altitude, value => {
      layer.altitude = value
    })
    assign(layer.height, layerPreset.height, value => {
      layer.height = value
    })
    assign(layer.densityScale, layerPreset.densityScale, value => {
      layer.densityScale = value
    })
    if (layerPreset.shapeAmount != null) {
      assign(layer.shapeAmount, layerPreset.shapeAmount, value => {
        layer.shapeAmount = value
      })
    }
    assign(layer.shapeDetailAmount, layerPreset.shapeDetailAmount, value => {
      layer.shapeDetailAmount = value
    })
  })

  assign(cloudsContext.coverage, scene.coverage, value => {
    cloudsContext.coverage = value
  })
  assign(
    cloudsContext.scatteringCoefficient,
    scene.scatteringCoefficient,
    value => {
      cloudsContext.scatteringCoefficient = value
    }
  )
  assign(
    cloudsContext.absorptionCoefficient,
    scene.absorptionCoefficient,
    value => {
      cloudsContext.absorptionCoefficient = value
    }
  )
  assign(cloudsContext.skyLightScale, scene.skyLightScale, value => {
    cloudsContext.skyLightScale = value
  })
  assign(cloudsContext.groundBounceScale, scene.groundBounceScale, value => {
    cloudsContext.groundBounceScale = value
  })
  if (
    cloudsContext.localWeatherRepeat.x !== scene.localWeatherRepeat ||
    cloudsContext.localWeatherRepeat.y !== scene.localWeatherRepeat
  ) {
    cloudsContext.localWeatherRepeat.setScalar(scene.localWeatherRepeat)
    changed = true
  }
  if (
    cloudsContext.shapeDetailRepeat.x !== scene.shapeDetailRepeat ||
    cloudsContext.shapeDetailRepeat.y !== scene.shapeDetailRepeat ||
    cloudsContext.shapeDetailRepeat.z !== scene.shapeDetailRepeat
  ) {
    cloudsContext.shapeDetailRepeat.setScalar(scene.shapeDetailRepeat)
    changed = true
  }
  assign(
    cloudsContext.turbulenceDisplacement,
    scene.turbulenceDisplacement,
    value => {
      cloudsContext.turbulenceDisplacement = value
    }
  )

  if (changed) {
    cloudsContext.invalidateHistory()
  }
}
