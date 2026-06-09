import { OrbitControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import {
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FC
} from 'react'
import {
  AgXToneMapping,
  Matrix4,
  Quaternion,
  Scene,
  Vector3,
  type Mesh
} from 'three'
import {
  context,
  mix,
  mrt,
  normalView,
  output,
  pass,
  toneMapping,
  uniform,
  uv,
  viewportUV,
  vec2,
  vec3,
  vec4
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
  AtmosphereContext
} from '@takram/three-atmosphere/webgpu'
import { CloudLayers, type CloudsQualityPreset } from '@takram/three-clouds'
import {
  clouds,
  cloudsShadow,
  cloudsShadowLength,
  CloudsContext
} from '@takram/three-clouds/webgpu'
import { Clouds as WebGPUCloudsContext } from '@takram/three-clouds/webgpu/r3f'
import {
  Ellipsoid,
  Geodetic,
  PointOfView,
  radians
} from '@takram/three-geospatial'
import { EllipsoidMesh } from '@takram/three-geospatial/r3f'
import {
  dithering,
  highpVelocity,
  lensFlare,
  stationaryDithering
} from '@takram/three-geospatial/webgpu'

import localWeatherUrl from '../../../packages/clouds/assets/local_weather.png?url'
import shapeDetailUrl from '../../../packages/clouds/assets/shape_detail.bin?url'
import shapeUrl from '../../../packages/clouds/assets/shape.bin?url'
import turbulenceUrl from '../../../packages/clouds/assets/turbulence.png?url'
import stbnUrl from '../../../packages/core/assets/stbn.bin?url'
import { applyCloudStoryPreset } from '../clouds/storyPresets'
import type { StoryFC } from '../components/createStory'
import { Description, TilesAttribution } from '../components/Description'
import { Globe } from '../components/Globe'
import { GlobeControls } from '../components/GlobeControls'
import { WebGPUCanvas } from '../components/WebGPUCanvas'
import {
  localDateArgs,
  localDateArgTypes,
  useLocalDateControls,
  type LocalDateArgs
} from '../controls/localDateControls'
import {
  outputPassArgs,
  outputPassArgTypes,
  useOutputPassControls,
  type OutputPassArgs
} from '../controls/outputPassControls'
import { rendererArgs, rendererArgTypes } from '../controls/rendererControls'
import {
  toneMappingArgs,
  toneMappingArgTypes,
  useToneMappingControls,
  type ToneMappingArgs
} from '../controls/toneMappingControls'
import {
  applyCameraMatrixToCamera,
  serializeCameraComponents,
  serializeCameraMatrixElements
} from '../helpers/cameraMatrixURL'
import {
  applyPointOfViewToCamera,
  readPointOfViewFromCamera
} from '../helpers/cameraPointOfView'
import { useControl } from '../hooks/useControl'
import { useGuardedFrame } from '../hooks/useGuardedFrame'
import type { PointOfViewProps } from '../hooks/usePointOfView'
import { useResource } from '../hooks/useResource'
import { useTransientControl } from '../hooks/useTransientControl'
import {
  applyFujiNoTilesCloudPreset,
  FUJI_PARITY_ANCHOR_LATITUDE,
  FUJI_PARITY_ANCHOR_LONGITUDE
} from './fujiParityPreset'

const LOCAL_WEATHER_VELOCITY = [0.001, 0] as const
const SHAPE_VELOCITY = [0.00012, 0, 0] as const
const SHAPE_DETAIL_VELOCITY = [0.0015, 0, 0] as const
const WEBGL_AERIAL_ALBEDO_SCALE = 2 / Math.PI
const WORLD_UP = new Vector3(0, 1, 0)
const targetScratch = new Vector3()
const eyeScratch = new Vector3()
const targetWorldScratch = new Vector3()
const cameraWorldScratch = new Vector3()
const upScratch = new Vector3()
const worldUpScratch = new Vector3()
const pointOfViewQuaternionScratch = new Quaternion()
const matrixScratch = new Matrix4()

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

function applyWebGLLikeMarchBudget(
  cloudsContext: CloudsContext,
  cameraFar: number = 4e5
): void {
  Object.assign(cloudsContext.clouds, {
    maxIterationCount: 500,
    minStepSize: 50,
    maxStepSize: 1000,
    maxRayDistance: 2e5,
    perspectiveStepScale: 1.01,
    minDensity: 1e-5,
    minExtinction: 1e-5,
    minTransmittance: 1e-2,
    maxIterationCountToGround: 3,
    maxIterationCountToSun: 12,
    minSecondaryStepSize: 100,
    secondaryStepScale: 2,
    maxShadowFilterRadius: 6,
    maxShadowLengthIterationCount: 500,
    minShadowLengthStepSize: 50,
    maxShadowLengthRayDistance: 2e5,
    hazeDensityScale: 3e-5,
    hazeExponent: 1e-3,
    hazeScatteringCoefficient: 0.9,
    hazeAbsorptionCoefficient: 0.8
  })
  cloudsContext.shadow.farScale = 0.25
  cloudsContext.shadow.maxFar = cameraFar * cloudsContext.shadow.farScale
  cloudsContext.skyLightScale = 0.79
  cloudsContext.scatterAnisotropy1 = 0.58
}

interface StoryProps extends PointOfViewProps {
  alignWithWebGLBasic?: boolean
  disableCloudStoryPreset?: boolean
  useFujiNoTilesCloudPreset?: boolean
  forceWebglLikeMarchBudget?: boolean
  disableFallbackNoApiKeyCameraOverride?: boolean
  disableFallbackEllipsoid?: boolean
  useIdentityWorldToECEFFrame?: boolean
  disableTiles?: boolean
  enableOrbitControls?: boolean
  orbitControlsTarget?: [number, number, number]
  cameraFar?: number
  hideDescription?: boolean
  cameraMatrixElements?: number[] | null
  cameraComponents?: number[] | null
  onCameraMatrixChange?: (elements: number[]) => void
  onCameraComponentsChange?: (components: number[]) => void
  alignWithWebGLLightingModel?: boolean
  enableCloudShadowAtlas?: boolean
}

type DebugRenderStage = 'final' | 'base' | 'aerial' | 'cloud'

interface StoryArgs extends OutputPassArgs, ToneMappingArgs, LocalDateArgs {
  googleMapsApiKey: string
  correctAltitude: boolean
  coverage: number
  qualityPreset: CloudsQualityPreset
  resolutionScale: number
  taaEnabled: boolean
  discardAllHistory: boolean
  velocityThresholdPixels: number
  historyResetDistanceThreshold: number
  temporalAlpha: number
  temporalUpscale: boolean
  temporalUpscaleScale: number
  animateClouds: boolean
  cloudMotionScale: number
  shapeDetail: boolean
  turbulence: boolean
  haze: boolean
  debugRenderStage?: DebugRenderStage
  debugDisableAerialLighting?: boolean
  debugDisableGeometricCorrection?: boolean
  debugDisableAerialNormal?: boolean
  debugFreezeLocalDate?: boolean
}

const Content: FC<StoryProps> = ({
  longitude,
  latitude,
  height,
  heading,
  pitch,
  distance,
  alignWithWebGLBasic = false,
  disableCloudStoryPreset = false,
  useFujiNoTilesCloudPreset = false,
  forceWebglLikeMarchBudget = false,
  disableFallbackNoApiKeyCameraOverride = false,
  disableFallbackEllipsoid = false,
  useIdentityWorldToECEFFrame = false,
  disableTiles = false,
  alignWithWebGLLightingModel = false,
  enableCloudShadowAtlas = false,
  cameraFar,
  cameraMatrixElements = null,
  cameraComponents = null,
  onCameraMatrixChange,
  onCameraComponentsChange
}) => {
  const renderer = useThree<Renderer>(({ gl }) => gl as any)
  const scene = useThree(({ scene }) => scene)
  const camera = useThree(({ camera }) => camera)
  const overlayScene = useMemo(() => new Scene(), [])
  const ellipsoidRef = useRef<Mesh>(null)
  const appliedCameraMatrixSignatureRef = useRef<string>('')
  const emittedCameraMatrixSignatureRef = useRef<string>('')
  const appliedCameraComponentsSignatureRef = useRef<string>('')
  const emittedCameraComponentsSignatureRef = useRef<string>('')

  const atmosphereContext = useResource(() => {
    const context = new AtmosphereContext()
    if (alignWithWebGLLightingModel && cameraFar != null && cameraFar > 1e6) {
      context.parameters.groundAlbedo.setScalar(0.25)
    }
    if (alignWithWebGLLightingModel) {
      // Match the WebGL aerial-perspective lookup path and avoid frame-varying
      // STBN raymarch noise in static parity captures.
      context.raymarchScattering = false
    }
    return context
  }, [alignWithWebGLLightingModel, cameraFar])
  atmosphereContext.camera = camera
  const cloudsShadowNode = useResource(() => cloudsShadow(camera), [camera])

  const cloudsContext = useResource(() => {
    const context = new CloudsContext()
    context.cloudLayers.copy(CloudLayers.DEFAULT)
    if (!alignWithWebGLBasic && !disableCloudStoryPreset) {
      applyCloudStoryPreset(context, 'ground')
    } else if (useFujiNoTilesCloudPreset) {
      applyFujiNoTilesCloudPreset(context)
    }
    applyCloudMotion(context, false, 1)
    return context
  }, [alignWithWebGLBasic, disableCloudStoryPreset, useFujiNoTilesCloudPreset])

  const apiKey = useControl(({ googleMapsApiKey }: StoryArgs) =>
    !disableTiles && googleMapsApiKey !== '' ? googleMapsApiKey : undefined
  )
  const temporalUpscale = useControl(
    ({ temporalUpscale }: StoryArgs) => temporalUpscale
  )
  const correctAltitude = useControl(
    ({ correctAltitude }: StoryArgs) => correctAltitude
  )
  const coverage = useControl(({ coverage }: StoryArgs) => coverage)
  const qualityPreset = useControl(
    ({ qualityPreset }: StoryArgs) => qualityPreset
  )
  const resolutionScale = useControl(
    ({ resolutionScale }: StoryArgs) => resolutionScale
  )
  const taaEnabled = useControl(({ taaEnabled }: StoryArgs) => taaEnabled)
  const discardAllHistory = useControl(
    ({ discardAllHistory }: StoryArgs) => discardAllHistory
  )
  const velocityThresholdPixels = useControl(
    ({ velocityThresholdPixels }: StoryArgs) => velocityThresholdPixels
  )
  const historyResetDistanceThreshold = useControl(
    ({ historyResetDistanceThreshold }: StoryArgs) =>
      historyResetDistanceThreshold
  )
  const temporalAlpha = useControl(
    ({ temporalAlpha }: StoryArgs) => temporalAlpha
  )
  const temporalUpscaleScale = useControl(
    ({ temporalUpscaleScale }: StoryArgs) => temporalUpscaleScale
  )
  const animateClouds = useControl(
    ({ animateClouds }: StoryArgs) => animateClouds
  )
  const cloudMotionScale = useControl(
    ({ cloudMotionScale }: StoryArgs) => cloudMotionScale
  )
  const shapeDetail = useControl(({ shapeDetail }: StoryArgs) => shapeDetail)
  const turbulence = useControl(({ turbulence }: StoryArgs) => turbulence)
  const haze = useControl(({ haze }: StoryArgs) => haze)
  const debugRenderStage = useControl(
    ({ debugRenderStage = 'final' }: StoryArgs) => debugRenderStage
  )
  const debugDisableAerialLighting = useControl(
    ({ debugDisableAerialLighting = false }: StoryArgs) =>
      debugDisableAerialLighting
  )
  const debugDisableGeometricCorrection = useControl(
    ({ debugDisableGeometricCorrection = false }: StoryArgs) =>
      debugDisableGeometricCorrection
  )
  const debugDisableAerialNormal = useControl(
    ({ debugDisableAerialNormal = false }: StoryArgs) =>
      debugDisableAerialNormal
  )
  const debugFreezeLocalDate = useControl(
    ({ debugFreezeLocalDate = false }: StoryArgs) => debugFreezeLocalDate
  )
  const debugDayOfYear = useControl(({ dayOfYear }: StoryArgs) => dayOfYear)
  const debugTimeOfDay = useControl(({ timeOfDay }: StoryArgs) => timeOfDay)
  const debugYear = useControl(({ year }: StoryArgs) => year)
  const useStableHighOrbitParity =
    alignWithWebGLLightingModel && cameraFar != null && cameraFar > 1e6
  const useCloudShadowAtlas = enableCloudShadowAtlas
  const useCloudShadowLength = enableCloudShadowAtlas && !useStableHighOrbitParity

  useLayoutEffect(() => {
    if (cameraFar == null) {
      return
    }
    const cameraWithProjection = camera as typeof camera & {
      far: number
      updateProjectionMatrix: () => void
    }
    if (cameraWithProjection.far === cameraFar) {
      return
    }
    cameraWithProjection.far = cameraFar
    cameraWithProjection.updateProjectionMatrix()
  }, [camera, cameraFar])

  useLayoutEffect(() => {
    if (apiKey != null) {
      return
    }

    if (alignWithWebGLBasic || useIdentityWorldToECEFFrame) {
      atmosphereContext.matrixWorldToECEF.value.identity()

      if (ellipsoidRef.current != null) {
        ellipsoidRef.current.matrix.identity()
        ellipsoidRef.current.matrixAutoUpdate = false
        ellipsoidRef.current.matrixWorldNeedsUpdate = true
        ellipsoidRef.current.updateMatrixWorld(true)
      }
      return
    }

    const anchorECEF = new Geodetic(
      radians(FUJI_PARITY_ANCHOR_LONGITUDE),
      radians(FUJI_PARITY_ANCHOR_LATITUDE),
      0
    ).toECEF(new Vector3())

    Ellipsoid.WGS84.getNorthUpEastFrame(
      anchorECEF,
      atmosphereContext.matrixWorldToECEF.value
    )

    if (ellipsoidRef.current == null) {
      return
    }

    ellipsoidRef.current.matrix
      .copy(atmosphereContext.matrixWorldToECEF.value)
      .invert()
    ellipsoidRef.current.matrixAutoUpdate = false
    ellipsoidRef.current.matrixWorldNeedsUpdate = true
    ellipsoidRef.current.updateMatrixWorld(true)
  }, [
    alignWithWebGLBasic,
    useIdentityWorldToECEFFrame,
    apiKey,
    atmosphereContext
  ])

  useLayoutEffect(() => {
    atmosphereContext.correctAltitude = correctAltitude
    cloudsContext.qualityPreset = qualityPreset
    if (!alignWithWebGLBasic && !disableCloudStoryPreset) {
      applyCloudStoryPreset(cloudsContext, 'ground')
    }
    if (forceWebglLikeMarchBudget) {
      applyWebGLLikeMarchBudget(cloudsContext, cameraFar)
    }
    const baseCoverage = alignWithWebGLBasic
      ? Math.min(coverage * (5 / 3), 1)
      : coverage
    cloudsContext.coverage = baseCoverage
    cloudsContext.resolutionScale = resolutionScale
    cloudsContext.temporalAntialias = taaEnabled
    cloudsContext.discardAllHistory = discardAllHistory
    cloudsContext.velocityThresholdPixels = velocityThresholdPixels
    cloudsContext.historyResetDistanceThreshold = historyResetDistanceThreshold
    cloudsContext.temporalAlpha = temporalAlpha < 0 ? null : temporalAlpha
    // Keep static comparison captures deterministic; moving clouds still animate
    // blue-noise samples so temporal accumulation can converge over time.
    const animateStbn = taaEnabled && animateClouds
    cloudsContext.animateStbn = animateStbn
    if (!animateStbn) {
      cloudsContext.stbnFrameIndex = 0
    }
    cloudsContext.temporalUpscaleScale = temporalUpscaleScale
    applyCloudMotion(cloudsContext, animateClouds, cloudMotionScale)
    cloudsContext.shapeDetail = shapeDetail
    cloudsContext.turbulence = turbulence
    cloudsContext.haze = haze
  }, [
    alignWithWebGLBasic,
    animateClouds,
    atmosphereContext,
    cloudMotionScale,
    cloudsContext,
    correctAltitude,
    coverage,
    discardAllHistory,
    disableCloudStoryPreset,
    forceWebglLikeMarchBudget,
    haze,
    historyResetDistanceThreshold,
    cameraFar,
    qualityPreset,
    resolutionScale,
    shapeDetail,
    taaEnabled,
    temporalAlpha,
    temporalUpscaleScale,
    turbulence,
    velocityThresholdPixels
  ])

  const fallbackGlobeMaterial = useResource(() => {
    if (alignWithWebGLLightingModel) {
      return new MeshBasicNodeMaterial({
        color: '#6b695e'
      })
    }
    return new MeshBasicNodeMaterial({
      colorNode: mix(
        vec3(0.18, 0.19, 0.16),
        vec3(0.42, 0.4, 0.34),
        uv().y.pow(1.55)
      )
    })
  }, [alignWithWebGLLightingModel])

  const passNode = useResource(
    () =>
      pass(scene, camera, { samples: 4 }).setMRT(
        mrt({
          output,
          normal: normalView,
          velocity: highpVelocity
        })
      ),
    [scene, camera]
  )

  const colorNode = passNode.getTextureNode('output')
  const depthNode = passNode.getTextureNode('depth')
  const normalNode = passNode.getTextureNode('normal')
  const shadowLengthNode = useResource(
    () => cloudsShadowLength(depthNode, camera, cloudsShadowNode),
    [depthNode, camera, cloudsShadowNode]
  )

  useLayoutEffect(() => {
    cloudsShadowNode.setContexts(cloudsContext, atmosphereContext)
    shadowLengthNode.setContexts(cloudsContext, atmosphereContext)
    renderer.contextNode = context({
      ...renderer.contextNode.value,
      getAtmosphere: () => atmosphereContext,
      getClouds: () => cloudsContext,
      getCloudsShadow: useCloudShadowAtlas
        ? () => cloudsShadowNode
        : undefined,
      getCloudsShadowLength: useCloudShadowLength
        ? () => shadowLengthNode
        : undefined
    })
  }, [
    renderer,
    atmosphereContext,
    cloudsContext,
    cloudsShadowNode,
    shadowLengthNode,
    useCloudShadowAtlas,
    useCloudShadowLength
  ])

  const enableWebGLLightingModel =
    alignWithWebGLLightingModel && !debugDisableAerialLighting

  const aerialInputNode = useResource(
    () =>
      enableWebGLLightingModel
        ? vec4(colorNode.rgb.mul(WEBGL_AERIAL_ALBEDO_SCALE), colorNode.a)
        : colorNode,
    [enableWebGLLightingModel, colorNode]
  )

  const aerialNode = useResource(
    () =>
      aerialPerspective(
        aerialInputNode,
        depthNode,
        useCloudShadowLength
          ? vec2(shadowLengthNode.sampleShadowLength(viewportUV), 0)
          : null
      ),
    [
      aerialInputNode,
      cloudsShadowNode,
      shadowLengthNode,
      depthNode,
      useCloudShadowLength,
      normalNode
    ]
  )

  const cloudNode = useResource(() => {
    cloudsContext.temporalUpscale = temporalUpscale
    return clouds(aerialNode, depthNode, camera)
  }, [
    aerialNode,
    camera,
    cloudsContext,
    cloudsShadowNode,
    shadowLengthNode,
    depthNode,
    useCloudShadowAtlas,
    useCloudShadowLength,
    taaEnabled,
    temporalUpscale
  ])
  const lensFlareNode = useResource(() => lensFlare(cloudNode), [cloudNode])

  const toneMappingNode = useResource(
    () => toneMapping(AgXToneMapping, uniform(1), lensFlareNode),
    [lensFlareNode]
  )
  const ditherNode = useMemo(
    () => (useStableHighOrbitParity ? stationaryDithering : dithering),
    [useStableHighOrbitParity]
  )

  const overlayPassNode = useResource(
    () =>
      pass(overlayScene, camera, {
        samples: 0,
        depthBuffer: false
      }),
    [camera, overlayScene]
  )

  const finalNode = useResource(
    () =>
      toneMappingNode
        .add(ditherNode)
        .mul(overlayPassNode.a.oneMinus())
        .add(overlayPassNode),
    [toneMappingNode, overlayPassNode, ditherNode]
  )

  const postProcessing = useResource(
    () => {
      const outputNode =
        debugRenderStage === 'base'
          ? colorNode
          : debugRenderStage === 'aerial'
            ? aerialNode
            : debugRenderStage === 'cloud'
              ? cloudNode
              : finalNode
      return new RenderPipeline(renderer, outputNode)
    },
    [
      renderer,
      colorNode,
      aerialNode,
      cloudNode,
      finalNode,
      debugRenderStage
    ]
  )

  useLayoutEffect(() => {
    aerialNode.lighting = enableWebGLLightingModel
    aerialNode.correctGeometricError = !debugDisableGeometricCorrection
    aerialNode.normalNode = debugDisableAerialNormal ? null : normalNode
    postProcessing.needsUpdate = true
  }, [
    aerialNode,
    debugDisableAerialNormal,
    debugDisableGeometricCorrection,
    enableWebGLLightingModel,
    normalNode,
    postProcessing
  ])

  useGuardedFrame(() => {
    postProcessing.render()
  }, 1)

  useOutputPassControls(
    postProcessing,
    passNode,
    (outputNode, outputColorTransform) => {
      postProcessing.outputNode = outputNode
      postProcessing.outputColorTransform = outputColorTransform
      postProcessing.needsUpdate = true
    }
  )

  useToneMappingControls(toneMappingNode, () => {
    postProcessing.needsUpdate = true
  })

  const useLocalFramePointOfView =
    apiKey == null && !alignWithWebGLBasic && !useIdentityWorldToECEFFrame

  useLayoutEffect(() => {
    if (alignWithWebGLBasic || useLocalFramePointOfView) {
      return
    }

    new PointOfView(distance, radians(heading), radians(pitch)).decompose(
      new Geodetic(radians(longitude), radians(latitude), height ?? 0).toECEF(
        targetScratch
      ),
      camera.position,
      camera.quaternion,
      camera.up
    )
    camera.updateMatrixWorld()
  }, [
    alignWithWebGLBasic,
    useLocalFramePointOfView,
    longitude,
    latitude,
    height,
    heading,
    pitch,
    distance,
    camera
  ])

  useLayoutEffect(() => {
    if (!useLocalFramePointOfView) {
      return
    }

    const targetECEF = new Geodetic(
      radians(longitude),
      radians(latitude),
      height ?? 0
    ).toECEF(targetScratch)

    const ecefToWorld = matrixScratch
      .copy(atmosphereContext.matrixWorldToECEF.value)
      .invert()
    const targetWorld = targetWorldScratch
      .copy(targetECEF)
      .applyMatrix4(ecefToWorld)
    new PointOfView(distance, radians(heading), radians(pitch)).decompose(
      targetECEF,
      eyeScratch,
      pointOfViewQuaternionScratch,
      upScratch
    )
    const cameraWorld = cameraWorldScratch
      .copy(eyeScratch)
      .applyMatrix4(ecefToWorld)
    const upWorld = worldUpScratch
      .copy(upScratch)
      .transformDirection(ecefToWorld)
      .normalize()

    camera.position.copy(cameraWorld)
    camera.up.copy(upWorld)
    camera.lookAt(targetWorld)
    camera.updateMatrixWorld()
  }, [
    useLocalFramePointOfView,
    longitude,
    latitude,
    height,
    heading,
    pitch,
    distance,
    camera,
    atmosphereContext
  ])

  useLayoutEffect(() => {
    if (!alignWithWebGLBasic) {
      return
    }

    const target = new Geodetic(
      radians(longitude),
      radians(latitude),
      height ?? 0
    ).toECEF(new Vector3())
    const up = Ellipsoid.WGS84.getSurfaceNormal(target, new Vector3())
    const rotation = new Quaternion().setFromUnitVectors(WORLD_UP, up)
    const offset = new Vector3(0, 0, 5).applyQuaternion(rotation)

    camera.up.copy(up)
    camera.position.copy(target).add(offset)
    camera.lookAt(target)
    camera.updateMatrixWorld()
  }, [alignWithWebGLBasic, camera, height, latitude, longitude])

  useLayoutEffect(() => {
    if (
      apiKey != null ||
      disableTiles ||
      alignWithWebGLBasic ||
      disableFallbackNoApiKeyCameraOverride
    ) {
      return
    }

    camera.position.set(0, 1800, -1500)
    camera.up.set(0, 1, 0)
    camera.lookAt(36000, 2300, 1800)
    camera.updateMatrixWorld()
  }, [
    alignWithWebGLBasic,
    disableFallbackNoApiKeyCameraOverride,
    apiKey,
    camera
  ])

  useLayoutEffect(() => {
    if (cameraComponents == null || cameraComponents.length !== 6) {
      return
    }
    const signature = serializeCameraComponents(cameraComponents)
    if (signature === appliedCameraComponentsSignatureRef.current) {
      return
    }
    if (
      !applyPointOfViewToCamera(
        camera,
        cameraComponents,
        atmosphereContext.matrixWorldToECEF.value
      )
    ) {
      return
    }
    appliedCameraComponentsSignatureRef.current = signature
    emittedCameraComponentsSignatureRef.current = signature
  }, [atmosphereContext, camera, cameraComponents])

  useLayoutEffect(() => {
    if (cameraComponents != null && cameraComponents.length === 6) {
      return
    }
    if (cameraMatrixElements == null || cameraMatrixElements.length !== 16) {
      return
    }
    const signature = serializeCameraMatrixElements(cameraMatrixElements)
    if (signature === appliedCameraMatrixSignatureRef.current) {
      return
    }
    applyCameraMatrixToCamera(camera, cameraMatrixElements)
    appliedCameraMatrixSignatureRef.current = signature
    emittedCameraMatrixSignatureRef.current = signature
  }, [camera, cameraMatrixElements])

  useFrame(() => {
    if (onCameraComponentsChange != null) {
      const components = readPointOfViewFromCamera(
        camera,
        atmosphereContext.matrixWorldToECEF.value
      )
      if (components != null) {
        const signature = serializeCameraComponents(components)
        if (signature !== emittedCameraComponentsSignatureRef.current) {
          emittedCameraComponentsSignatureRef.current = signature
          appliedCameraComponentsSignatureRef.current = signature
          onCameraComponentsChange(components.slice(0, 6))
        }
      }
    }
    if (onCameraMatrixChange != null) {
      const elements = camera.matrixWorld.elements.slice(0, 16)
      const signature = serializeCameraMatrixElements(elements)
      if (signature !== emittedCameraMatrixSignatureRef.current) {
        emittedCameraMatrixSignatureRef.current = signature
        appliedCameraMatrixSignatureRef.current = signature
        onCameraMatrixChange(elements)
      }
    }
  }, 2)

  const updateAtmosphereDate = useCallback(
    (date: number) => {
      const { matrixECIToECEF, sunDirectionECEF, moonDirectionECEF } =
        atmosphereContext
      getECIToECEFRotationMatrix(date, matrixECIToECEF.value)
      getSunDirectionECI(date, sunDirectionECEF.value).applyMatrix4(
        matrixECIToECEF.value
      )
      getMoonDirectionECI(date, moonDirectionECEF.value).applyMatrix4(
        matrixECIToECEF.value
      )
    },
    [atmosphereContext]
  )

  useLocalDateControls(longitude, date => {
    if (debugFreezeLocalDate) {
      return
    }
    updateAtmosphereDate(date)
  })

  useLayoutEffect(() => {
    if (!debugFreezeLocalDate) {
      return
    }
    const epoch = Date.UTC(debugYear, 0, 1, 0, 0, 0, 0)
    const offset = longitude / 15
    updateAtmosphereDate(
      epoch +
        ((Math.floor(debugDayOfYear) - 1) * 24 +
          debugTimeOfDay -
          offset) *
          3600000
    )
  }, [
    debugDayOfYear,
    debugFreezeLocalDate,
    debugTimeOfDay,
    debugYear,
    longitude,
    updateAtmosphereDate
  ])

  const cloudsProvider = (
    <WebGPUCloudsContext
      context={cloudsContext}
      localWeatherTexture={localWeatherUrl}
      shapeTexture={shapeUrl}
      shapeDetailTexture={shapeDetailUrl}
      turbulenceTexture={turbulenceUrl}
      stbnTexture={stbnUrl}
    />
  )

  if (apiKey == null || disableTiles) {
    if (alignWithWebGLBasic || disableFallbackEllipsoid) {
      return <>{cloudsProvider}</>
    }
    return (
      <>
        {cloudsProvider}
        <EllipsoidMesh
          ref={ellipsoidRef}
          args={[Ellipsoid.WGS84.radii, 192, 96]}
          material={fallbackGlobeMaterial}
        />
      </>
    )
  }

  return (
    <>
      {cloudsProvider}
      <Globe apiKey={apiKey}>
        <GlobeControls enableDamping overlayScene={overlayScene} />
      </Globe>
    </>
  )
}

export const Story: StoryFC<StoryProps, StoryArgs> = props => (
  <WebGPUCanvas
    camera={{
      near: 1,
      far: props.cameraFar ?? 4e7
    }}
    renderer={{
      logarithmicDepthBuffer: true
    }}
    showBackendMessage={!props.hideDescription}
  >
    <Content {...props} />
    {props.enableOrbitControls && (
      <OrbitControls
        enableDamping
        target={props.orbitControlsTarget ?? [0, 0, 0]}
      />
    )}
    {!props.hideDescription && (
      <Description>
        <p>
          This viewer is intended as the WebGPU comparison page for the legacy
          WebGL clouds viewer. It keeps atmosphere, sun direction from local
          date, and volumetric clouds in one page so the WebGPU path can be
          compared under similar viewing conditions.
        </p>
        <p>
          In <code>alignWithWebGLBasic</code> mode, coverage is remapped
          internally for visual parity (0.3 → 0.5 equivalent in the current
          implementation).
        </p>
        <p>
          Google Photorealistic 3D Tiles require a valid Google Maps API key
          only when tiles are enabled. If the terrain does not appear, provide{' '}
          <code>google maps api key</code> from the controls panel or configure{' '}
          <code>STORYBOOK_GOOGLE_MAP_API_KEY</code>.
        </p>
        <p>
          Some variants (including no-tiles comparisons) disable 3D Tiles
          entirely and do not require a Google Maps API key.
        </p>
        <TilesAttribution />
      </Description>
    )}
  </WebGPUCanvas>
)

Story.args = {
  googleMapsApiKey: '',
  correctAltitude: true,
  coverage: 0.3,
  qualityPreset: 'high',
  resolutionScale: 1,
  taaEnabled: true,
  discardAllHistory: false,
  velocityThresholdPixels: 6,
  historyResetDistanceThreshold: 100,
  temporalAlpha: -1,
  temporalUpscale: false,
  temporalUpscaleScale: 0.375,
  animateClouds: false,
  cloudMotionScale: 1,
  shapeDetail: true,
  turbulence: true,
  haze: true,
  debugRenderStage: 'final',
  debugDisableAerialLighting: false,
  debugDisableGeometricCorrection: false,
  debugDisableAerialNormal: false,
  debugFreezeLocalDate: false,
  ...localDateArgs({
    dayOfYear: 260,
    timeOfDay: 16
  }),
  ...toneMappingArgs({
    toneMappingExposure: 3.9
  }),
  ...outputPassArgs(),
  ...rendererArgs()
}

Story.argTypes = {
  googleMapsApiKey: { control: 'text' },
  correctAltitude: {
    control: {
      type: 'boolean'
    },
    table: { category: 'atmosphere' }
  },
  coverage: {
    control: {
      type: 'range',
      min: 0,
      max: 1,
      step: 0.01
    },
    table: { category: 'clouds' }
  },
  qualityPreset: {
    control: {
      type: 'select'
    },
    options: ['low', 'medium', 'high', 'ultra'],
    table: { category: 'clouds' }
  },
  resolutionScale: {
    name: 'resolution scale',
    control: {
      type: 'range',
      min: 0.25,
      max: 1,
      step: 0.05
    },
    table: { category: 'rendering' }
  },
  taaEnabled: {
    name: 'taa enabled',
    control: {
      type: 'boolean'
    },
    table: { category: 'rendering' }
  },
  discardAllHistory: {
    name: 'discard all history',
    control: {
      type: 'boolean'
    },
    table: { category: 'rendering' }
  },
  velocityThresholdPixels: {
    name: 'velocity threshold (px)',
    control: {
      type: 'range',
      min: 0,
      max: 24,
      step: 0.25
    },
    table: { category: 'rendering' }
  },
  historyResetDistanceThreshold: {
    name: 'history reset distance (m)',
    control: {
      type: 'range',
      min: 0,
      max: 5000,
      step: 10
    },
    table: { category: 'rendering' }
  },
  temporalAlpha: {
    name: 'temporal alpha (-1 auto)',
    control: {
      type: 'range',
      min: -1,
      max: 1,
      step: 0.01
    },
    table: { category: 'rendering' }
  },
  temporalUpscale: {
    name: 'temporal upscale',
    control: {
      type: 'boolean'
    },
    table: { category: 'rendering' }
  },
  temporalUpscaleScale: {
    name: 'temporal upscale scale',
    control: {
      type: 'range',
      min: 0.0625,
      max: 1,
      step: 0.0625
    },
    table: { category: 'rendering' }
  },
  animateClouds: {
    name: 'animate clouds',
    control: {
      type: 'boolean'
    },
    table: { category: 'clouds' }
  },
  cloudMotionScale: {
    name: 'cloud motion',
    control: {
      type: 'range',
      min: 0,
      max: 4,
      step: 0.05
    },
    table: { category: 'clouds' }
  },
  shapeDetail: {
    name: 'shape detail',
    control: {
      type: 'boolean'
    },
    table: { category: 'rendering' }
  },
  turbulence: {
    control: {
      type: 'boolean'
    },
    table: { category: 'rendering' }
  },
  haze: {
    control: {
      type: 'boolean'
    },
    table: { category: 'rendering' }
  },
  debugRenderStage: {
    name: 'debug render stage',
    control: {
      type: 'select'
    },
    options: ['final', 'base', 'aerial', 'cloud'] satisfies DebugRenderStage[],
    table: { category: 'debug' }
  },
  debugDisableAerialLighting: {
    name: 'disable aerial lighting',
    control: {
      type: 'boolean'
    },
    table: { category: 'debug' }
  },
  debugDisableGeometricCorrection: {
    name: 'disable geometric correction',
    control: {
      type: 'boolean'
    },
    table: { category: 'debug' }
  },
  debugDisableAerialNormal: {
    name: 'disable aerial normal',
    control: {
      type: 'boolean'
    },
    table: { category: 'debug' }
  },
  debugFreezeLocalDate: {
    name: 'freeze local date',
    control: {
      type: 'boolean'
    },
    table: { category: 'debug' }
  },
  ...localDateArgTypes(),
  ...toneMappingArgTypes(),
  ...outputPassArgTypes(),
  ...rendererArgTypes()
}

export default Story
