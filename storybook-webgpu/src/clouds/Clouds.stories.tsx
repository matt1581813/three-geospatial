import type { Meta } from '@storybook/react-vite'

import { Story as ComparisonStory } from '../cloud/Cloud-Story'
import { createStory } from '../components/createStory'
import { Story } from './Clouds-Story'

import ComparisonCode from '../cloud/Cloud-Story?raw'
import Code from './Clouds-Story?raw'

export default {
  title: 'clouds/WebGPU Clouds',
  tags: ['order:3'],
  parameters: {
    docs: {
      codePanel: true,
      source: {
        language: 'tsx'
      }
    }
  }
} satisfies Meta

export const GroundBaseline = createStory(Story, {
  args: {
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1
  },
  argTypes: {
    animateClouds: {
      control: false,
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

export const ComparisonViewer = createStory(ComparisonStory, {
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
        code: ComparisonCode
      }
    }
  }
})

export const WebGLvsWebGPUParity = createStory(ComparisonStory, {
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
        code: ComparisonCode
      }
    }
  }
})

export const GroundAnimated = createStory(Story, {
  args: {
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    animateClouds: true
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})

export const GroundBeerShadows = createStory(Story, {
  args: {
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    cloudShadowAtlas: true
  },
  argTypes: {
    animateClouds: {
      control: false,
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

export const GroundBeerShadowsLowSun = createStory(Story, {
  args: {
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    cloudShadowAtlas: true,
    sunPreset: 'lowSun',
    toneMappingExposure: 5.2
  },
  argTypes: {
    animateClouds: {
      control: false,
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

export const GroundLightShaftsGlobal = createStory(Story, {
  args: {
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: true,
    cloudShadowAtlas: true,
    lightShafts: true,
    shadowLengthPass: true,
    animateStbn: false,
    sunPreset: 'lowSun',
    toneMappingExposure: 5.2
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})

export const GroundTemporalUpscale = createStory(Story, {
  args: {
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: true,
    animateStbn: true
  },
  argTypes: {
    animateClouds: {
      control: false,
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

export const GroundTemporalUpscaleAnimated = createStory(Story, {
  args: {
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

export const CruiseView = createStory(Story, {
  args: {
    viewPreset: 'cruise',
    qualityPreset: 'high',
    resolutionScale: 1
  },
  argTypes: {
    animateClouds: {
      control: false,
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

export const GroundTemporalUpscaleCameraCut = createStory(Story, {
  args: {
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: true,
    animateClouds: true,
    animateStbn: true,
    cloudMotionScale: 1,
    scriptedCameraCut: true
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})

export const GroundLightShaftsCameraCut = createStory(Story, {
  args: {
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: true,
    cloudShadowAtlas: true,
    lightShafts: true,
    animateClouds: true,
    animateStbn: true,
    cloudMotionScale: 1,
    scriptedCameraCut: true
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})

export const GroundTemporalUpscaleResizeReset = createStory(Story, {
  args: {
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: true,
    animateClouds: true,
    animateStbn: true,
    cloudMotionScale: 1,
    scriptedResize: true
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})

export const GroundLightShaftsResizeReset = createStory(Story, {
  args: {
    viewPreset: 'ground',
    qualityPreset: 'high',
    resolutionScale: 1,
    temporalUpscale: true,
    cloudShadowAtlas: true,
    lightShafts: true,
    animateClouds: true,
    animateStbn: true,
    cloudMotionScale: 1,
    scriptedResize: true
  },
  parameters: {
    docs: {
      source: {
        code: Code
      }
    }
  }
})
