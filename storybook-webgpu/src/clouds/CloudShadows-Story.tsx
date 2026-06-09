import { OrbitControls } from '@react-three/drei'
import { extend, useThree, type ThreeElement } from '@react-three/fiber'
import {
  useEffect,
  useLayoutEffect,
  useMemo,
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
  PerspectiveCamera,
  RedFormat,
  RepeatWrapping,
  TextureLoader,
  Vector3,
  type Camera,
  type Mesh,
  type Texture
} from 'three'
import {
  context,
  float,
  mix,
  normalWorld,
  pass,
  positionWorld,
  remapClamp,
  toneMapping,
  uniform,
  uv,
  vec3
} from 'three/tsl'
import {
  MeshBasicNodeMaterial,
  RenderPipeline,
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
  cloudShadow,
  clouds,
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

import localWeatherUrl from '../../../packages/clouds/assets/local_weather.png?url'
import shapeDetailUrl from '../../../packages/clouds/assets/shape_detail.bin?url'
import shapeUrl from '../../../packages/clouds/assets/shape.bin?url'
import turbulenceUrl from '../../../packages/clouds/assets/turbulence.png?url'
import stbnUrl from '../../../packages/core/assets/stbn.bin?url'
import {
  applyCloudStoryPreset,
  type CloudStoryViewPreset
} from './storyPresets'

const ANCHOR_LONGITUDE = 138.7274
const ANCHOR_LATITUDE = 35.3606
const SCENE_DATE = new Date('2025-09-21T06:00:00Z')
const LOCAL_WEATHER_VELOCITY = [0.00035, 0] as const
const SHAPE_VELOCITY = [0.00012, 0, 0] as const
const SHAPE_DETAIL_VELOCITY = [0.0015, 0, 0] as const

declare module '@react-three/fiber' {
  interface ThreeElements {
    atmosphereLight: ThreeElement<typeof AtmosphereLight>
  }
}

extend({ AtmosphereLight })

function useLoadTexture(url: string): Texture | null {
  const [texture, setTexture] = useState<Texture | null>(null)

  useEffect(() => {
    let active = true
    const loaded = new TextureLoader().load(url, nextTexture => {
      nextTexture.minFilter = LinearMipMapLinearFilter
      nextTexture.magFilter = LinearFilter
      nextTexture.wrapS = RepeatWrapping
      nextTexture.wrapT = RepeatWrapping
      nextTexture.colorSpace = NoColorSpace
      nextTexture.needsUpdate = true
      if (active) {
        setTexture(nextTexture)
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
    }).load(url, nextTexture => {
      if (active) {
        setTexture(nextTexture)
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
    const loaded = new STBNLoader().load(url, nextTexture => {
      if (active) {
        setTexture(nextTexture)
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

function applySunDate(atmosphere: AtmosphereContext): void {
  const { matrixECIToECEF, sunDirectionECEF, moonDirectionECEF } = atmosphere
  getECIToECEFRotationMatrix(SCENE_DATE, matrixECIToECEF.value)
  getSunDirectionECI(SCENE_DATE, sunDirectionECEF.value).applyMatrix4(
    matrixECIToECEF.value
  )
  getMoonDirectionECI(SCENE_DATE, moonDirectionECEF.value).applyMatrix4(
    matrixECIToECEF.value
  )
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

function applyShadowStoryPreset(
  cloudsContext: CloudsContext,
  viewPreset: ViewPreset
): void {
  if (viewPreset === 'ground') {
    cloudsContext.shadow.maxFar = 80_000
    cloudsContext.shadow.margin = 6_000
    cloudsContext.shadow.splitLambda = 0.72
    cloudsContext.shadow.mapSize.setScalar(1_536)
    cloudsContext.shadow.maxIterationCount = 72
    cloudsContext.shadow.minStepSize = 60
    cloudsContext.shadow.maxStepSize = 700
    cloudsContext.shadow.minTransmittance = 5e-5
    cloudsContext.shadow.opticalDepthTailScale = 2.4
    cloudsContext.coverage = 0.28
    cloudsContext.localWeatherRepeat.setScalar(10)
    cloudsContext.cloudLayers[0].shadow = true
    cloudsContext.cloudLayers[0].altitude = 2_200
    cloudsContext.cloudLayers[0].height = 1_700
    cloudsContext.cloudLayers[0].densityScale = 0.2
    cloudsContext.cloudLayers[0].shapeAmount = 0.72
    cloudsContext.cloudLayers[0].shapeDetailAmount = 0.82
    cloudsContext.cloudLayers[1].shadow = true
    cloudsContext.cloudLayers[1].altitude = 4_800
    cloudsContext.cloudLayers[1].height = 1_000
    cloudsContext.cloudLayers[1].densityScale = 0.03
    cloudsContext.cloudLayers[1].shapeAmount = 0.42
    cloudsContext.cloudLayers[1].shapeDetailAmount = 0.2
    cloudsContext.cloudLayers[2].shadow = false
    cloudsContext.cloudLayers[2].densityScale = 0.001
    cloudsContext.cloudLayers[2].shapeAmount = 0.2
    cloudsContext.cloudLayers[2].shapeDetailAmount = 0
    return
  }

  cloudsContext.shadow.maxFar = 180_000
  cloudsContext.shadow.margin = 12_000
  cloudsContext.shadow.splitLambda = 0.6
  cloudsContext.shadow.mapSize.setScalar(1_024)
  cloudsContext.shadow.maxIterationCount = 50
  cloudsContext.shadow.minStepSize = 100
  cloudsContext.shadow.maxStepSize = 1_000
  cloudsContext.shadow.minTransmittance = 1e-4
  cloudsContext.shadow.opticalDepthTailScale = 2
  cloudsContext.coverage = 0.52
  cloudsContext.localWeatherRepeat.setScalar(20)
  cloudsContext.cloudLayers[0].densityScale = 0.18
  cloudsContext.cloudLayers[1].densityScale = 0.2
  cloudsContext.cloudLayers[2].densityScale = 0.0015
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

function getViewPose(
  viewPreset: ViewPreset
): {
  position: [number, number, number]
  target: [number, number, number]
  receiver: [number, number, number]
} {
  if (viewPreset === 'ground') {
    return {
      position: [4_000, 8_500, -12_000],
      target: [36_000, 1_200, 1_800],
      receiver: [36_000, -4, 1_800]
    }
  }

  return {
    position: [8_000, 14_000, -18_000],
    target: [52_000, 5_300, 2_200],
    receiver: [52_000, -4, 2_200]
  }
}

type ViewPreset = CloudStoryViewPreset

interface StoryProps {}

interface StoryArgs extends RendererArgs, ToneMappingArgs {
  qualityPreset: CloudsQualityPreset
  resolutionScale: number
  temporalUpscaleScale: number
  temporalUpscale: boolean
  animateClouds: boolean
  animateStbn: boolean
  cloudMotionScale: number
  shadowStrength: number
  debugAtlas: boolean
  viewPreset: ViewPreset
}

const SceneContent: FC<StoryProps> = () => {
  const renderer = useThree<Renderer>(({ gl }) => gl as any)
  const scene = useThree(({ scene }) => scene)
  const camera = useThree(({ camera }) => camera)
  const ellipsoidRef = useRef<Mesh>(null)
  const controlsRef = useRef<ComponentRef<typeof OrbitControls>>(null)
  const shadowDebugCamera = useMemo(() => new PerspectiveCamera(), [])

  const atmosphereContext = useResource(() => new AtmosphereContext(), [])
  atmosphereContext.camera = camera

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
    applyCloudStoryPreset(context, 'ground')
    applyShadowStoryPreset(context, 'ground')
    applyCloudMotion(context, false, 1)
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
    applySunDate(atmosphereContext)
    if (ellipsoidRef.current != null) {
      ellipsoidRef.current.matrix
        .copy(atmosphereContext.matrixWorldToECEF.value)
        .invert()
      ellipsoidRef.current.matrixAutoUpdate = false
      ellipsoidRef.current.matrixWorldNeedsUpdate = true
      ellipsoidRef.current.updateMatrixWorld(true)
    }
  }, [atmosphereContext])

  useLayoutEffect(() => {
    renderer.contextNode = context({
      ...renderer.contextNode.value,
      getAtmosphere: () => atmosphereContext,
      getClouds: () => cloudsContext
    })
  }, [renderer, atmosphereContext, cloudsContext])

  useLayoutEffect(() => {
    scene.backgroundNode = null
    scene.environmentNode = null
    return () => {
      scene.backgroundNode = null
      scene.environmentNode = null
    }
  }, [scene])

  const qualityPreset = useControl(({ qualityPreset }: StoryArgs) => qualityPreset)
  const resolutionScale = useControl(
    ({ resolutionScale }: StoryArgs) => resolutionScale
  )
  const temporalUpscaleScale = useControl(
    ({ temporalUpscaleScale }: StoryArgs) => temporalUpscaleScale
  )
  const temporalUpscale = useControl(
    ({ temporalUpscale }: StoryArgs) => temporalUpscale
  )
  const animateClouds = useControl(({ animateClouds }: StoryArgs) => animateClouds)
  const animateStbn = useControl(({ animateStbn }: StoryArgs) => animateStbn)
  const cloudMotionScale = useControl(
    ({ cloudMotionScale }: StoryArgs) => cloudMotionScale
  )
  const shadowStrength = useControl(({ shadowStrength }: StoryArgs) => shadowStrength)
  const debugAtlas = useControl(({ debugAtlas }: StoryArgs) => debugAtlas)
  const viewPreset = useControl(({ viewPreset }: StoryArgs) => viewPreset)

  useTransientControl(
    (args: StoryArgs) => ({
      qualityPreset: args.qualityPreset,
      resolutionScale: args.resolutionScale,
      temporalUpscaleScale: args.temporalUpscaleScale,
      temporalUpscale: args.temporalUpscale,
      animateClouds: args.animateClouds,
      animateStbn: args.animateStbn,
      cloudMotionScale: args.cloudMotionScale,
      viewPreset: args.viewPreset
    }),
    (value, previousValue) => {
      cloudsContext.qualityPreset = value.qualityPreset
      if (previousValue == null || value.viewPreset !== previousValue.viewPreset) {
        applyCloudStoryPreset(cloudsContext, value.viewPreset)
        applyShadowStoryPreset(cloudsContext, value.viewPreset)
      }
      cloudsContext.resolutionScale = value.resolutionScale
      cloudsContext.temporalUpscaleScale = value.temporalUpscaleScale
      cloudsContext.temporalUpscale = value.temporalUpscale
      cloudsContext.animateStbn = value.animateStbn
      applyCloudMotion(cloudsContext, value.animateClouds, value.cloudMotionScale)
    }
  )

  useLayoutEffect(() => {
    const pose = getViewPose(viewPreset)

    if (debugAtlas) {
      applyCameraPose(camera, [0, 0, 18], [0, 0, 0], null)
      if ((camera as PerspectiveCamera).isPerspectiveCamera === true) {
        shadowDebugCamera.copy(camera as PerspectiveCamera, false)
      }
      if ((camera as PerspectiveCamera).isPerspectiveCamera === true) {
        const mainPerspective = camera as PerspectiveCamera
        shadowDebugCamera.fov = mainPerspective.fov
        shadowDebugCamera.near = mainPerspective.near
        shadowDebugCamera.far = mainPerspective.far
        shadowDebugCamera.aspect = mainPerspective.aspect
        shadowDebugCamera.zoom = mainPerspective.zoom
        shadowDebugCamera.updateProjectionMatrix()
      }
      applyCameraPose(
        shadowDebugCamera,
        pose.position,
        pose.target,
        null
      )
      return
    }

    applyCameraPose(camera, pose.position, pose.target, controlsRef.current)
  }, [camera, controlsRef, debugAtlas, viewPreset, shadowDebugCamera])

  const shadowNode = useResource(
    () => cloudsShadow(debugAtlas ? shadowDebugCamera : camera),
    [camera, debugAtlas, shadowDebugCamera]
  )
  shadowNode.setContexts(cloudsContext, atmosphereContext)

  const passNode = useResource(
    () => pass(scene, camera, { samples: 4 }),
    [scene, camera]
  )
  const colorNode = passNode.getTextureNode('output')
  const depthNode = passNode.getTextureNode('depth')
  const aerialNode = useResource(
    () => aerialPerspective(colorNode, depthNode),
    [colorNode, depthNode]
  )
  const cloudsNode = useResource(
    () => {
      cloudsContext.temporalUpscale = temporalUpscale
      return clouds(aerialNode, depthNode, camera)
    },
    [aerialNode, depthNode, camera, cloudsContext, temporalUpscale]
  )
  const toneMappingNode = useResource(
    () => toneMapping(AgXToneMapping, uniform(1), cloudsNode),
    [cloudsNode]
  )
  const postProcessing = useResource(
    () => new RenderPipeline(renderer, toneMappingNode),
    [renderer, toneMappingNode]
  )

  useToneMappingControls(toneMappingNode, () => {
    postProcessing.needsUpdate = true
  })

  useGuardedFrame(() => {
    if (debugAtlas) {
      renderer.render(scene, camera)
      return
    }
    postProcessing.render()
  }, 1)

  const groundShadowNode = useResource(
    () => cloudShadow(positionWorld, { shadowNode, normalNode: normalWorld }),
    [shadowNode]
  )
  const groundShadowResponseNode = useResource(
    () =>
      remapClamp(
        float(1).sub(groundShadowNode.clamp(0, 1)),
        float(0.02),
        float(0.26)
      )
        .pow(0.8),
    [groundShadowNode]
  )

  const groundPatchMaterial = useResource(
    () =>
      new MeshBasicNodeMaterial({
        colorNode: vec3(0.44, 0.42, 0.38).mul(
          float(1)
            .sub(float(shadowStrength).clamp(0, 1).mul(groundShadowResponseNode))
            .clamp(0.08, 1)
        )
      }),
    [groundShadowResponseNode, shadowStrength]
  )

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

  const atlasMaterial = useResource(
    () =>
      new MeshBasicNodeMaterial({
        // Atlas stores [frontDepth, meanExtinction, maxOpticalDepth, tailDepth].
        // Visualize B+A (optical depth) instead of R (distance), otherwise
        // the debug view collapses to near-black.
        colorNode: vec3(
          remapClamp(
            shadowNode.getTextureNode().b.add(shadowNode.getTextureNode().a),
            float(0),
            float(0.2)
          )
        )
      }),
    [shadowNode]
  )

  if (debugAtlas) {
    return (
      <mesh>
        <planeGeometry args={[14, 4.5, 1, 1]} />
        <primitive object={atlasMaterial} attach='material' />
      </mesh>
    )
  }

  return (
    <>
      <mesh
        position={getViewPose(viewPreset).receiver}
        rotation={[-Math.PI / 2, 0, 0]}
      >
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
  const debugAtlas = useControl(({ debugAtlas }: StoryArgs) => debugAtlas)

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        maxWidth: 'min(100%, 1400px)',
        aspectRatio: debugAtlas ? '16 / 9' : '16 / 9',
        minHeight: 480,
        margin: '0 auto'
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
          fov: debugAtlas ? 34 : 38,
          near: 0.1,
          far: 1e9
        }}
      >
        <SceneContent {...props} />
        <Description>
          <p>
            WebGPU cloud shadows render a shadow atlas through{' '}
            <code>CloudsShadowNode</code>, then ground receivers explicitly sample
            it with <code>cloudShadow(positionWorld, ...)</code>.
          </p>
          <p>
            This is the first M3 slice: stable single-frame Beer shadow maps on a
            cascade atlas. It does not include shadow temporal resolve or light
            shafts yet.
          </p>
          <p>
            The receiver path is explicit on the material node graph. There is no
            global material injection in this slice.
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
  temporalUpscale: true,
  animateClouds: false,
  animateStbn: true,
  cloudMotionScale: 1,
  shadowStrength: 0.8,
  debugAtlas: false,
  viewPreset: 'ground',
  ...toneMappingArgs({
    toneMappingExposure: 3.9
  }),
  ...rendererArgs()
}

Story.argTypes = {
  qualityPreset: {
    control: { type: 'select' },
    options: ['low', 'medium', 'high', 'ultra']
  },
  resolutionScale: {
    name: 'marching resolution scale',
    control: { type: 'range', min: 0.25, max: 1, step: 0.05 }
  },
  temporalUpscaleScale: {
    name: 'temporal upscale scale',
    control: { type: 'range', min: 0.0625, max: 1, step: 0.0625 }
  },
  temporalUpscale: {
    name: 'temporal upscale',
    control: { type: 'boolean' }
  },
  animateClouds: {
    name: 'animate clouds',
    control: { type: 'boolean' }
  },
  animateStbn: {
    name: 'animate STBN',
    control: { type: 'boolean' }
  },
  cloudMotionScale: {
    name: 'cloud motion',
    control: { type: 'range', min: 0, max: 4, step: 0.05 }
  },
  shadowStrength: {
    name: 'shadow strength',
    control: { type: 'range', min: 0, max: 0.9, step: 0.05 }
  },
  debugAtlas: {
    name: 'debug atlas',
    control: { type: 'boolean' }
  },
  viewPreset: {
    name: 'view preset',
    control: { type: 'radio' },
    options: ['ground', 'cruise']
  },
  ...toneMappingArgTypes(),
  ...rendererArgTypes()
}

export default Story
