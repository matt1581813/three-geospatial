import type { Meta } from '@storybook/react-vite'

import { createStory } from '../components/createStory'
import { Story } from './CloudShadows-Story'

import Code from './CloudShadows-Story?raw'

export default {
  title: 'clouds/WebGPU Cloud Shadows',
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

export const GroundCloudShadows = createStory(Story, {
  args: {
    debugAtlas: false,
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: true,
    animateClouds: false,
    animateStbn: true
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})

export const GroundCloudShadowsAnimated = createStory(Story, {
  args: {
    debugAtlas: false,
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: true,
    animateClouds: true,
    animateStbn: true,
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

export const GroundCloudShadowsDebugAtlas = createStory(Story, {
  args: {
    debugAtlas: true,
    qualityPreset: 'high',
    animateClouds: true,
    animateStbn: true,
    temporalUpscale: false
  },
  argTypes: {
    shadowStrength: {
      table: {
        disable: true
      }
    },
    viewPreset: {
      table: {
        disable: true
      }
    }
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})
