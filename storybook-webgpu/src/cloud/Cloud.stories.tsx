import type { Meta } from '@storybook/react-vite'

import { createStory } from '../components/createStory'
import { Story } from './Cloud-Story'
import { Story as FujiParityStory } from './FujiParity-Story'
import { FUJI_PARITY_POINT_OF_VIEW } from './fujiParityPreset'

import Code from './Cloud-Story?raw'
import FujiParityCode from './FujiParity-Story?raw'

export default {
  title: 'cloud',
  tags: ['order:4'],
  parameters: {
    docs: {
      codePanel: true,
      source: {
        language: 'tsx'
      }
    }
  }
} satisfies Meta

export const Cloud = createStory(Story, {
  props: {
    longitude: 138.5973,
    latitude: 35.2138,
    heading: 71,
    pitch: -31,
    distance: 7000
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})

export const WebGLBasicAligned = createStory(Story, {
  props: {
    alignWithWebGLBasic: true,
    longitude: 30,
    latitude: 35,
    height: 300,
    heading: -69.3659,
    pitch: 0,
    distance: 5
  },
  args: {
    dayOfYear: 1,
    timeOfDay: 9,
    toneMappingExposure: 10,
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: true,
    temporalUpscaleScale: 0.375,
    coverage: 0.3,
    correctAltitude: true,
    shapeDetail: true,
    turbulence: true,
    haze: true,
    animateClouds: false,
    cloudMotionScale: 1
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})

export const WebGLFujiAligned = createStory(Story, {
  props: {
    disableTiles: true,
    disableCloudStoryPreset: true,
    useFujiNoTilesCloudPreset: true,
    disableFallbackNoApiKeyCameraOverride: true,
    disableFallbackEllipsoid: true,
    useIdentityWorldToECEFFrame: false,
    longitude: 138.634,
    latitude: 35.5,
    heading: -91,
    pitch: -27,
    distance: 8444
  },
  args: {
    dayOfYear: 200,
    timeOfDay: 17.5,
    toneMappingExposure: 10,
    coverage: 0.48,
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: false,
    temporalUpscaleScale: 0.375,
    correctAltitude: true,
    shapeDetail: true,
    turbulence: true,
    haze: true,
    animateClouds: false,
    cloudMotionScale: 1
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})

export const WebGLFujiTilesParity = createStory(Story, {
  props: {
    disableTiles: false,
    disableCloudStoryPreset: true,
    useFujiNoTilesCloudPreset: false,
    disableFallbackNoApiKeyCameraOverride: true,
    disableFallbackEllipsoid: true,
    useIdentityWorldToECEFFrame: true,
    longitude: 138.634,
    latitude: 35.5,
    heading: -91,
    pitch: -27,
    distance: 8444
  },
  args: {
    dayOfYear: 200,
    timeOfDay: 17.5,
    toneMappingExposure: 10,
    coverage: 0.4,
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: false,
    temporalUpscaleScale: 0.375,
    correctAltitude: true,
    shapeDetail: true,
    turbulence: true,
    haze: true,
    animateClouds: true,
    cloudMotionScale: 1
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})

export const FujiParity = createStory(FujiParityStory, {
  props: {
    ...FUJI_PARITY_POINT_OF_VIEW
  },
  args: {
    cloudPresetMode: 'legacy-default',
    animateClouds: true
  },
  parameters: {
    docs: {
      source: {
        code: FujiParityCode
      }
    }
  }
})
