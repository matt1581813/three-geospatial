import { OrbitControls } from '@react-three/drei'
import { extend, useThree, type ThreeElement } from '@react-three/fiber'
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ComponentRef,
  type FC
} from 'react'
import {
  AgXToneMapping,
  Data3DTexture,
  LinearFilter,
  LinearMipMapLinearFilter,
  NoColorSpace,
  RedFormat,
  RepeatWrapping,
  TextureLoader,
  Vector3,
  type Camera,
  type Mesh,
  type Texture
} from 'three'
import { context, mix, pass, toneMapping, uniform, uv, vec3 } from 'three/tsl'
import {
  MeshBasicNodeMaterial,
  PostProcessing,
  type Renderer
} from 'three/webgpu'

import {
  getECIToECEFRotationMatrix,
  getMoonDirectionECI,
  getSunDirectionECI
} from '@takram/three-atmosphere'
import {
  aerialPerspective,
  AtmosphereContext,
  AtmosphereLight,
  AtmosphereLightNode
} from '@takram/three-atmosphere/webgpu'
import {
  CLOUD_SHAPE_DETAIL_TEXTURE_SIZE,
  CLOUD_SHAPE_TEXTURE_SIZE,
  CloudLayers,
  type CloudsQualityPreset
} from '@takram/three-clouds'
import {
  clouds,
  cloudsShadowLength,
  cloudsShadow,
  CloudsContext
} from '@takram/three-clouds/webgpu'
import {
  DataTextureLoader,
  Ellipsoid,
  Geodetic,
  parseUint8Array,
  radians,
  STBNLoader
} from '@takram/three-geospatial'
import { EllipsoidMesh } from '@takram/three-geospatial/r3f'

import type { StoryFC } from '../components/createStory'
import { Description } from '../components/Description'
import { WebGPUCanvas } from '../components/WebGPUCanvas'
import {
  rendererArgs,
  rendererArgTypes,
  type RendererArgs
} from '../controls/rendererControls'
import {
  toneMappingArgs,
  toneMappingArgTypes,
  useToneMappingControls,
  type ToneMappingArgs
} from '../controls/toneMappingControls'
import { useControl } from '../hooks/useControl'
import { useGuardedFrame } from '../hooks/useGuardedFrame'
import { useResource } from '../hooks/useResource'
import { useTransientControl } from '../hooks/useTransientControl'
import {
  applyCloudStoryPreset,
  type CloudStoryViewPreset
} from './storyPresets'

import localWeatherUrl from '../../../packages/clouds/assets/local_weather.png?url'
import shapeDetailUrl from '../../../packages/clouds/assets/shape_detail.bin?url'
import shapeUrl from '../../../packages/clouds/assets/shape.bin?url'
import turbulenceUrl from '../../../packages/clouds/assets/turbulence.png?url'
import stbnUrl from '../../../packages/core/assets/stbn.bin?url'

const ANCHOR_LONGITUDE = 138.7274
const ANCHOR_LATITUDE = 35.3606
const SCENE_DATES = {
  day: new Date('2025-09-21T06:00:00Z'),
  lowSun: new Date('2025-09-21T08:00:00Z')
} as const
const LOCAL_WEATHER_VELOCITY = [0.00035, 0] as const
const SHAPE_VELOCITY = [0.00012, 0, 0] as const
const SHAPE_DETAIL_VELOCITY = [0.0015, 0, 0] as const
const CAMERA_CUT_INTERVAL = 2_800
const RESIZE_SEQUENCE = [1, 0.76, 1, 0.68, 0.9, 1] as const
const RESIZE_INTERVAL = 2_400

declare module '@react-three/fiber' {
  interface ThreeElements {
    atmosphereLight: ThreeElement<typeof AtmosphereLight>
  }
}

extend({ AtmosphereLight })

function loadTexture(url: string): Texture {
  return new TextureLoader().load(url, texture => {
    texture.minFilter = LinearMipMapLinearFilter
    texture.magFilter = LinearFilter
    texture.wrapS = RepeatWrapping
    texture.wrapT = RepeatWrapping
    texture.colorSpace = NoColorSpace
    texture.needsUpdate = true
  })
}

function load3DTexture(url: string, size: number): Data3DTexture {
  return new DataTextureLoader(Data3DTexture, parseUint8Array, {
    width: size,
    height: size,
    depth: size,
    format: RedFormat,
    minFilter: LinearFilter,
    magFilter: LinearFilter,
    wrapS: RepeatWrapping,
    wrapT: RepeatWrapping,
    wrapR: RepeatWrapping,
    colorSpace: NoColorSpace
  }).load(url)
}

function useLoadTexture(url: string): Texture | null {
  const [texture, setTexture] = useState<Texture | null>(null)

  useEffect(() => {
    let active = true
    const loaded = new TextureLoader().load(url, texture => {
      texture.minFilter = LinearMipMapLinearFilter
      texture.magFilter = LinearFilter
      texture.wrapS = RepeatWrapping
      texture.wrapT = RepeatWrapping
      texture.colorSpace = NoColorSpace
      texture.needsUpdate = true
      if (active) {
        setTexture(texture)
      }
    })
    return () => {
      active = false
      setTexture(null)
      loaded.dispose()
    }
  }, [url])

  return texture
}

function useLoad3DTexture(url: string, size: number): Data3DTexture | null {
  const [texture, setTexture] = useState<Data3DTexture | null>(null)

  useEffect(() => {
    let active = true
    const loaded = new DataTextureLoader(Data3DTexture, parseUint8Array, {
      width: size,
      height: size,
      depth: size,
      format: RedFormat,
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      wrapS: RepeatWrapping,
      wrapT: RepeatWrapping,
      wrapR: RepeatWrapping,
      colorSpace: NoColorSpace
    }).load(url, texture => {
      if (active) {
        setTexture(texture)
      }
    })
    return () => {
      active = false
      setTexture(null)
      loaded.dispose()
    }
  }, [url, size])

  return texture
}

function useLoadStbnTexture(url: string): Data3DTexture | null {
  const [texture, setTexture] = useState<Data3DTexture | null>(null)

  useEffect(() => {
    let active = true
    const loaded = new STBNLoader().load(url, texture => {
      if (active) {
        setTexture(texture)
      }
    })
    return () => {
      active = false
      setTexture(null)
      loaded.dispose()
    }
  }, [url])

  return texture
}

type SunPreset = keyof typeof SCENE_DATES

function applySunDate(
  atmosphere: AtmosphereContext,
  sunPreset: SunPreset
): void {
  const sceneDate = SCENE_DATES[sunPreset]
  const { matrixECIToECEF, sunDirectionECEF, moonDirectionECEF } = atmosphere
  getECIToECEFRotationMatrix(sceneDate, matrixECIToECEF.value)
  getSunDirectionECI(sceneDate, sunDirectionECEF.value).applyMatrix4(
    matrixECIToECEF.value
  )
  getMoonDirectionECI(sceneDate, moonDirectionECEF.value).applyMatrix4(
    matrixECIToECEF.value
  )
}

function getViewPresetPose(preset: ViewPreset): {
  position: [number, number, number]
  target: [number, number, number]
} {
  const position: [number, number, number] =
    preset === 'ground' ? [0, 1_800, -1_500] : [0, 5_600, -2_800]
  const target: [number, number, number] =
    preset === 'ground' ? [36_000, 2_300, 1_800] : [52_000, 5_300, 2_200]

  return { position, target }
}

function applyCameraPose(
  camera: Camera,
  position: [number, number, number],
  target: [number, number, number],
  controls?: ComponentRef<typeof OrbitControls> | null
): void {
  camera.position.fromArray(position)
  camera.up.set(0, 1, 0)
  camera.lookAt(target[0], target[1], target[2])
  camera.updateMatrixWorld()
  if (controls != null) {
    controls.target.set(target[0], target[1], target[2])
    controls.update()
  }
}

function getCameraCutPoses(
  preset: ViewPreset
): readonly {
  position: [number, number, number]
  target: [number, number, number]
}[] {
  const base = getViewPresetPose(preset)
  if (preset === 'ground') {
    return [
      base,
      {
        position: [16_000, 2_500, -17_000],
        target: [48_000, 4_400, 5_500]
      }
    ] as const
  }
  return [
    base,
    {
      position: [24_000, 6_900, -15_000],
      target: [63_000, 5_900, 4_200]
    }
  ] as const
}

function applyCloudMotion(
  cloudsContext: CloudsContext,
  animateClouds: boolean,
  motionScale: number
): void {
  const scale = Math.max(motionScale, 0)
  if (animateClouds) {
    cloudsContext.localWeatherVelocity
      .fromArray(LOCAL_WEATHER_VELOCITY)
      .multiplyScalar(scale)
    cloudsContext.shapeVelocity.fromArray(SHAPE_VELOCITY).multiplyScalar(scale)
    cloudsContext.shapeDetailVelocity
      .fromArray(SHAPE_DETAIL_VELOCITY)
      .multiplyScalar(scale)
    return
  }
  cloudsContext.localWeatherVelocity.setScalar(0)
  cloudsContext.shapeVelocity.setScalar(0)
  cloudsContext.shapeDetailVelocity.setScalar(0)
}

function applyCloudShadowStoryPreset(
  cloudsContext: CloudsContext,
  viewPreset: ViewPreset
): void {
  let changed = false

  const assign = <T,>(
    target: T,
    next: T,
    apply: (value: T) => void
  ): void => {
    if (target === next) {
      return
    }
    apply(next)
    changed = true
  }

  assign(cloudsContext.cloudLayers[0].shadow, true, value => {
    cloudsContext.cloudLayers[0].shadow = value
  })
  assign(cloudsContext.cloudLayers[1].shadow, true, value => {
    cloudsContext.cloudLayers[1].shadow = value
  })
  assign(cloudsContext.cloudLayers[2].shadow, true, value => {
    cloudsContext.cloudLayers[2].shadow = value
  })
  if (viewPreset === 'ground') {
    assign(cloudsContext.shadow.maxFar, 80_000, value => {
      cloudsContext.shadow.maxFar = value
    })
    assign(cloudsContext.shadow.margin, 6_000, value => {
      cloudsContext.shadow.margin = value
    })
    assign(cloudsContext.shadow.splitLambda, 0.72, value => {
      cloudsContext.shadow.splitLambda = value
    })
    if (
      cloudsContext.shadow.mapSize.x !== 1_536 ||
      cloudsContext.shadow.mapSize.y !== 1_536
    ) {
      cloudsContext.shadow.mapSize.setScalar(1_536)
      changed = true
    }
    assign(cloudsContext.shadow.maxIterationCount, 72, value => {
      cloudsContext.shadow.maxIterationCount = value
    })
    assign(cloudsContext.shadow.minStepSize, 60, value => {
      cloudsContext.shadow.minStepSize = value
    })
    assign(cloudsContext.shadow.maxStepSize, 700, value => {
      cloudsContext.shadow.maxStepSize = value
    })
    assign(cloudsContext.shadow.minTransmittance, 5e-5, value => {
      cloudsContext.shadow.minTransmittance = value
    })
    assign(cloudsContext.shadow.opticalDepthTailScale, 2.4, value => {
      cloudsContext.shadow.opticalDepthTailScale = value
    })
    assign(cloudsContext.clouds.maxShadowLengthIterationCount, 16, value => {
      cloudsContext.clouds.maxShadowLengthIterationCount = value
    })
    assign(cloudsContext.clouds.minShadowLengthStepSize, 50, value => {
      cloudsContext.clouds.minShadowLengthStepSize = value
    })
    assign(cloudsContext.clouds.maxShadowLengthRayDistance, 2e5, value => {
      cloudsContext.clouds.maxShadowLengthRayDistance = value
    })
  } else {
    assign(cloudsContext.shadow.maxFar, 180_000, value => {
      cloudsContext.shadow.maxFar = value
    })
    assign(cloudsContext.shadow.margin, 12_000, value => {
      cloudsContext.shadow.margin = value
    })
    assign(cloudsContext.shadow.splitLambda, 0.6, value => {
      cloudsContext.shadow.splitLambda = value
    })
    if (
      cloudsContext.shadow.mapSize.x !== 1_024 ||
      cloudsContext.shadow.mapSize.y !== 1_024
    ) {
      cloudsContext.shadow.mapSize.setScalar(1_024)
      changed = true
    }
    assign(cloudsContext.shadow.maxIterationCount, 50, value => {
      cloudsContext.shadow.maxIterationCount = value
    })
    assign(cloudsContext.shadow.minStepSize, 100, value => {
      cloudsContext.shadow.minStepSize = value
    })
    assign(cloudsContext.shadow.maxStepSize, 1_000, value => {
      cloudsContext.shadow.maxStepSize = value
    })
    assign(cloudsContext.shadow.minTransmittance, 1e-4, value => {
      cloudsContext.shadow.minTransmittance = value
    })
    assign(cloudsContext.shadow.opticalDepthTailScale, 2, value => {
      cloudsContext.shadow.opticalDepthTailScale = value
    })
    assign(cloudsContext.clouds.maxShadowLengthIterationCount, 500, value => {
      cloudsContext.clouds.maxShadowLengthIterationCount = value
    })
    assign(cloudsContext.clouds.minShadowLengthStepSize, 50, value => {
      cloudsContext.clouds.minShadowLengthStepSize = value
    })
    assign(cloudsContext.clouds.maxShadowLengthRayDistance, 2e5, value => {
      cloudsContext.clouds.maxShadowLengthRayDistance = value
    })
  }

  if (changed) {
    cloudsContext.invalidateHistory()
  }
}

type ViewPreset = CloudStoryViewPreset

interface StoryProps {}

interface StoryArgs extends RendererArgs, ToneMappingArgs {
  qualityPreset: CloudsQualityPreset
  resolutionScale: number
  temporalUpscaleScale: number
  temporalUpscale: boolean
  sunPreset: SunPreset
  cloudShadowAtlas: boolean
  lightShafts: boolean
  shadowLengthPass: boolean
  animateClouds: boolean
  cloudMotionScale: number
  animateStbn: boolean
  shapeDetail: boolean
  turbulence: boolean
  haze: boolean
  viewPreset: ViewPreset
  scriptedCameraCut: boolean
  scriptedResize: boolean
}

const Content: FC<StoryProps> = () => {
  const renderer = useThree<Renderer>(({ gl }) => gl as any)
  const scene = useThree(({ scene }) => scene)
  const camera = useThree(({ camera }) => camera)
  const ellipsoidRef = useRef<Mesh>(null)
  const controlsRef = useRef<ComponentRef<typeof OrbitControls>>(null)

  const atmosphereContext = useResource(() => new AtmosphereContext(), [])
  atmosphereContext.camera = camera
  const shadowNode = useResource(() => cloudsShadow(camera), [camera])

  const localWeatherTexture = useLoadTexture(localWeatherUrl)
  const shapeTexture = useLoad3DTexture(
    shapeUrl,
    CLOUD_SHAPE_TEXTURE_SIZE
  )
  const shapeDetailTexture = useLoad3DTexture(
    shapeDetailUrl,
    CLOUD_SHAPE_DETAIL_TEXTURE_SIZE
  )
  const turbulenceTexture = useLoadTexture(turbulenceUrl)
  const stbnTexture = useLoadStbnTexture(stbnUrl)

  const cloudsContext = useResource(() => {
    const context = new CloudsContext()
    context.cloudLayers.copy(CloudLayers.DEFAULT)
    applyCloudMotion(context, false, 1)
    applyCloudStoryPreset(context, 'ground')
    return context
  }, [])

  useLayoutEffect(() => {
    if (
      localWeatherTexture == null ||
      shapeTexture == null ||
      shapeDetailTexture == null ||
      turbulenceTexture == null ||
      stbnTexture == null
    ) {
      return
    }
    cloudsContext.localWeatherTexture = localWeatherTexture
    cloudsContext.shapeTexture = shapeTexture
    cloudsContext.shapeDetailTexture = shapeDetailTexture
    cloudsContext.turbulenceTexture = turbulenceTexture
    cloudsContext.stbnTexture = stbnTexture
  }, [
    cloudsContext,
    localWeatherTexture,
    shapeTexture,
    shapeDetailTexture,
    turbulenceTexture,
    stbnTexture
  ])

  const globeMaterial = useResource(
    () =>
      new MeshBasicNodeMaterial({
        colorNode: mix(
          vec3(0.18, 0.19, 0.16),
          vec3(0.42, 0.4, 0.34),
          uv().y.pow(1.55)
        )
      }),
    []
  )
  const groundPatchMaterial = useResource(
    () =>
      new MeshBasicNodeMaterial({
        colorNode: mix(
          vec3(0.17, 0.18, 0.15),
          vec3(0.34, 0.32, 0.27),
          uv().y.pow(1.3)
        )
      }),
    []
  )
  const sunPreset = useControl(({ sunPreset }: StoryArgs) => sunPreset)

  useLayoutEffect(() => {
    const anchorECEF = new Geodetic(
      radians(ANCHOR_LONGITUDE),
      radians(ANCHOR_LATITUDE),
      0
    ).toECEF(new Vector3())
    Ellipsoid.WGS84.getNorthUpEastFrame(
      anchorECEF,
      atmosphereContext.matrixWorldToECEF.value
    )
    applySunDate(atmosphereContext, sunPreset)
    if (ellipsoidRef.current != null) {
      ellipsoidRef.current.matrix
        .copy(atmosphereContext.matrixWorldToECEF.value)
        .invert()
      ellipsoidRef.current.matrixAutoUpdate = false
      ellipsoidRef.current.matrixWorldNeedsUpdate = true
      ellipsoidRef.current.updateMatrixWorld(true)
    }
  }, [atmosphereContext, sunPreset])

  useLayoutEffect(() => {
    scene.backgroundNode = null
    scene.environmentNode = null
    return () => {
      scene.backgroundNode = null
      scene.environmentNode = null
    }
  }, [scene])

  const viewPreset = useControl(({ viewPreset }: StoryArgs) => viewPreset)
  const scriptedCameraCut = useControl(
    ({ scriptedCameraCut }: StoryArgs) => scriptedCameraCut
  )
  const temporalUpscale = useControl(
    ({ temporalUpscale }: StoryArgs) => temporalUpscale
  )
  const cloudShadowAtlas = useControl(
    ({ cloudShadowAtlas }: StoryArgs) => cloudShadowAtlas
  )
  const lightShafts = useControl(({ lightShafts }: StoryArgs) => lightShafts)
  const shadowLengthPass = useControl(
    ({ shadowLengthPass }: StoryArgs) => shadowLengthPass
  )

  useLayoutEffect(() => {
    const { position, target } = getViewPresetPose(viewPreset)
    applyCameraPose(camera, position, target, controlsRef.current)
  }, [camera, viewPreset])

  useEffect(() => {
    if (!scriptedCameraCut) {
      return
    }

    const poses = getCameraCutPoses(viewPreset)
    let index = 0
    const interval = window.setInterval(() => {
      index = (index + 1) % poses.length
      const pose = poses[index]
      applyCameraPose(camera, pose.position, pose.target, controlsRef.current)
    }, CAMERA_CUT_INTERVAL)

    return () => {
      window.clearInterval(interval)
      const { position, target } = getViewPresetPose(viewPreset)
      applyCameraPose(camera, position, target, controlsRef.current)
    }
  }, [camera, scriptedCameraCut, viewPreset])

  useTransientControl(
    ({
      qualityPreset,
      resolutionScale,
      temporalUpscaleScale,
      temporalUpscale,
      sunPreset,
      cloudShadowAtlas,
      lightShafts,
      shadowLengthPass,
      animateClouds,
      cloudMotionScale,
      animateStbn,
      viewPreset,
      shapeDetail,
      turbulence,
      haze
    }: StoryArgs) => ({
      qualityPreset,
      resolutionScale,
      temporalUpscaleScale,
      temporalUpscale,
      sunPreset,
      cloudShadowAtlas,
      lightShafts,
      shadowLengthPass,
      animateClouds,
      cloudMotionScale,
      animateStbn,
      viewPreset,
      shapeDetail,
      turbulence,
      haze
    }),
    (value, prevValue) => {
      cloudsContext.qualityPreset = value.qualityPreset
      if (
        prevValue == null ||
        value.viewPreset !== prevValue.viewPreset ||
        value.cloudShadowAtlas !== prevValue.cloudShadowAtlas
      ) {
        applyCloudStoryPreset(cloudsContext, value.viewPreset)
      }
      if (value.cloudShadowAtlas) {
        applyCloudShadowStoryPreset(cloudsContext, value.viewPreset)
      }
      if (prevValue == null || value.sunPreset !== prevValue.sunPreset) {
        cloudsContext.invalidateHistory()
      }
      cloudsContext.resolutionScale = value.resolutionScale
      cloudsContext.temporalUpscaleScale = value.temporalUpscaleScale
      applyCloudMotion(
        cloudsContext,
        value.animateClouds,
        value.cloudMotionScale
      )
      cloudsContext.lightShafts = value.lightShafts
      cloudsContext.animateStbn = value.animateStbn
      cloudsContext.shapeDetail = value.shapeDetail
      cloudsContext.turbulence = value.turbulence
      cloudsContext.haze = value.haze
    }
  )

  const passNode = useResource(
    () => pass(scene, camera, { samples: 4 }),
    [scene, camera]
  )

  const colorNode = passNode.getTextureNode('output')
  const depthNode = passNode.getTextureNode('depth')
  const shadowLengthNode = useResource(
    () => cloudsShadowLength(depthNode, camera, shadowNode),
    [depthNode, camera, shadowNode]
  )

  useLayoutEffect(() => {
    shadowNode.setContexts(cloudsContext, atmosphereContext)
    shadowLengthNode.setContexts(cloudsContext, atmosphereContext)
    renderer.contextNode = context({
      ...renderer.contextNode.value,
      getAtmosphere: () => atmosphereContext,
      getClouds: () => cloudsContext,
      getCloudsShadow: cloudShadowAtlas ? () => shadowNode : undefined,
      getCloudsShadowLength:
        cloudShadowAtlas && lightShafts && shadowLengthPass
          ? () => shadowLengthNode
          : undefined
    })
  }, [
    renderer,
    atmosphereContext,
    cloudsContext,
    cloudShadowAtlas,
    lightShafts,
    shadowLengthPass,
    shadowNode,
    shadowLengthNode
  ])

  const aerialNode = useResource(
    () => aerialPerspective(colorNode, depthNode),
    [
      colorNode,
      depthNode,
      cloudShadowAtlas,
      lightShafts,
      shadowLengthPass,
      shadowLengthNode
    ]
  )

  const cloudsNode = useResource(
    () => {
      cloudsContext.temporalUpscale = temporalUpscale
      return clouds(aerialNode, depthNode, camera)
    },
    [
      aerialNode,
      camera,
      cloudShadowAtlas,
      cloudsContext,
      depthNode,
      lightShafts,
      shadowLengthPass,
      shadowNode,
      shadowLengthNode,
      temporalUpscale
    ]
  )

  const toneMappingNode = useResource(
    () => toneMapping(AgXToneMapping, uniform(1), cloudsNode),
    [cloudsNode]
  )

  const postProcessing = useResource(
    () => new PostProcessing(renderer, toneMappingNode),
    [renderer, toneMappingNode]
  )

  useGuardedFrame(() => {
    postProcessing.render()
  }, 1)

  useToneMappingControls(toneMappingNode, () => {
    postProcessing.needsUpdate = true
  })

  return (
    <>
      <mesh position={[0, -4, 22_000]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[260_000, 220_000, 1, 1]} />
        <primitive object={groundPatchMaterial} attach='material' />
      </mesh>
      <OrbitControls
        ref={controlsRef}
        enableDamping
        minDistance={400}
        maxDistance={220_000}
      />
      <EllipsoidMesh
        ref={ellipsoidRef}
        args={[Ellipsoid.WGS84.radii, 192, 96]}
        material={globeMaterial}
      />
    </>
  )
}

export const Story: StoryFC<StoryProps, StoryArgs> = props => {
  const scriptedResize = useControl(
    ({ scriptedResize }: StoryArgs) => scriptedResize
  )
  const [resizeStep, setResizeStep] = useState(0)

  useEffect(() => {
    if (!scriptedResize) {
      setResizeStep(0)
      return
    }

    let index = 0
    setResizeStep(index)
    const interval = window.setInterval(() => {
      index = (index + 1) % RESIZE_SEQUENCE.length
      setResizeStep(index)
    }, RESIZE_INTERVAL)

    return () => {
      window.clearInterval(interval)
      setResizeStep(0)
    }
  }, [scriptedResize])

  const viewportScale = scriptedResize ? RESIZE_SEQUENCE[resizeStep] : 1

  return (
    <div
      style={{
        position: 'relative',
        width: `${viewportScale * 100}%`,
        maxWidth: 'min(100%, 1400px)',
        aspectRatio: '16 / 9',
        minHeight: 480,
        margin: '0 auto',
        transition: 'width 180ms ease-out'
      }}
    >
      <WebGPUCanvas
        style={{ width: '100%', height: '100%' }}
        renderer={{
          logarithmicDepthBuffer: true,
          onInit: renderer => {
            renderer.library.addLight(AtmosphereLightNode, AtmosphereLight)
          }
        }}
        camera={{
          fov: 38,
          near: 10,
          far: 1e9
        }}
      >
        <Content {...props} />
        <Description>
          <p>
            WebGPU clouds are injected through <em>renderer.contextNode</em>{' '}
            and composed as{' '}
            <code>
              pass -&gt; aerialPerspective -&gt; clouds -&gt; toneMapping
            </code>
            .
          </p>
          <p>
            M2 keeps the same pipeline and uses{' '}
            <code>temporal upscale</code> as{' '}
            <strong>low-resolution cloud marching plus TemporalAntialiasNode
            resolve</strong>
            . The <code>resolution scale</code> control changes only the cloud
            marching resolution, not the final output size.
          </p>
          <p>
            Use <code>GroundTemporalUpscale</code> as the still-quality
            reference. <code>GroundTemporalUpscaleAnimated</code> is the motion
            validation story for M2, not the still-image source of truth.
          </p>
          <p>
            Valid M2 acceptance requires a real WebGPU session. If the fallback
            banner appears, the result is running under WebGL2 and is not the
            source of truth for WebGPU validation.
          </p>
          <p>
            Enabling <code>cloud shadow atlas</code> feeds the WebGPU Beer
            shadow atlas back into the cloud lighting pass so direct sunlight on
            the cloud body is attenuated. That is a WebGL parity step, but it is
            not yet the full legacy shadow resolve pipeline.
          </p>
          <p>
            When both <code>cloud shadow atlas</code> and{' '}
            <code>light shafts</code> are enabled, the story also injects a
            cloud shadow-length node into <code>aerialPerspective</code> through
            <code>renderer.contextNode</code> for global scene-side shafts and
            cloud-shadow attenuation.
          </p>
          <p>
            Toggle <code>shadow length pass</code> for FPS A/B checks of this
            global shadow-length step without changing other cloud controls.
          </p>
          <p>
            This story uses a local north-up-east tangent frame anchored over
            Mount Fuji so the camera can stay near the world origin while the
            atmosphere and clouds still evaluate in ECEF space.
          </p>
        </Description>
      </WebGPUCanvas>
    </div>
  )
}

Story.args = {
  qualityPreset: 'high',
  resolutionScale: 1,
  temporalUpscaleScale: 0.375,
  temporalUpscale: false,
  sunPreset: 'day',
  cloudShadowAtlas: false,
  lightShafts: true,
  shadowLengthPass: true,
  animateClouds: false,
  cloudMotionScale: 1,
  animateStbn: false,
  shapeDetail: true,
  turbulence: true,
  haze: true,
  viewPreset: 'ground',
  scriptedCameraCut: false,
  scriptedResize: false,
  ...toneMappingArgs({
    toneMappingExposure: 3.9
  }),
  ...rendererArgs()
}

Story.argTypes = {
  qualityPreset: {
    control: {
      type: 'select'
    },
    options: ['low', 'medium', 'high', 'ultra']
  },
  resolutionScale: {
    name: 'marching resolution scale',
    control: {
      type: 'range',
      min: 0.25,
      max: 1,
      step: 0.05
    }
  },
  temporalUpscaleScale: {
    name: 'temporal upscale scale',
    control: {
      type: 'range',
      min: 0.0625,
      max: 1,
      step: 0.0625
    }
  },
  temporalUpscale: {
    name: 'temporal upscale',
    control: {
      type: 'boolean'
    }
  },
  sunPreset: {
    name: 'sun preset',
    control: {
      type: 'inline-radio'
    },
    options: ['day', 'lowSun']
  },
  cloudShadowAtlas: {
    name: 'cloud shadow atlas',
    control: {
      type: 'boolean'
    }
  },
  lightShafts: {
    name: 'light shafts',
    control: {
      type: 'boolean'
    }
  },
  shadowLengthPass: {
    name: 'shadow length pass',
    control: {
      type: 'boolean'
    }
  },
  animateClouds: {
    name: 'animate clouds',
    control: {
      type: 'boolean'
    }
  },
  cloudMotionScale: {
    name: 'cloud motion',
    control: {
      type: 'range',
      min: 0,
      max: 4,
      step: 0.05
    }
  },
  animateStbn: {
    name: 'animate STBN',
    control: {
      type: 'boolean'
    }
  },
  shapeDetail: {
    name: 'shape detail',
    control: {
      type: 'boolean'
    }
  },
  turbulence: {
    control: {
      type: 'boolean'
    }
  },
  haze: {
    control: {
      type: 'boolean'
    }
  },
  viewPreset: {
    name: 'view preset',
    control: {
      type: 'radio'
    },
    options: ['ground', 'cruise']
  },
  scriptedCameraCut: {
    name: 'scripted camera cut',
    control: {
      type: 'boolean'
    },
    table: { category: 'm2 acceptance' }
  },
  scriptedResize: {
    name: 'scripted resize',
    control: {
      type: 'boolean'
    },
    table: { category: 'm2 acceptance' }
  },
  ...toneMappingArgTypes(),
  ...rendererArgTypes()
}

export default Story
