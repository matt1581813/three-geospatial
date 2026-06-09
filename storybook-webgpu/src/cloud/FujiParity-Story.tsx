import { OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { EffectComposer, SMAA, ToneMapping } from '@react-three/postprocessing'
import type { GlobeControls as GlobeControlsImpl } from '3d-tiles-renderer'
import {
  GLTFExtensionsPlugin,
  GoogleCloudAuthPlugin,
  TileCompressionPlugin,
  TilesFadePlugin,
  UpdateOnChangePlugin
} from '3d-tiles-renderer/plugins'
import {
  GlobeControls as LegacyGlobeControls,
  TilesAttributionOverlay,
  TilesPlugin,
  TilesRenderer
} from '3d-tiles-renderer/r3f'
import { ToneMappingMode } from 'postprocessing'
import {
  Fragment,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ComponentRef,
  type FC,
  type ReactNode
} from 'react'
import {
  ACESFilmicToneMapping,
  AgXToneMapping,
  CineonToneMapping,
  LinearToneMapping,
  Matrix4,
  NoToneMapping,
  Quaternion,
  ReinhardToneMapping,
  Vector3,
  type ToneMapping as ThreeToneMapping
} from 'three'
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js'

import {
  AerialPerspective,
  Atmosphere,
  type AtmosphereApi
} from '@takram/three-atmosphere/r3f'
import {
  CloudLayers,
  type CloudsQualityPreset
} from '@takram/three-clouds'
import { CloudLayer, Clouds } from '@takram/three-clouds/r3f'
import {
  Ellipsoid,
  Geodetic,
  PointOfView,
  radians
} from '@takram/three-geospatial'
import { Dithering, LensFlare } from '@takram/three-geospatial-effects/r3f'
import { EllipsoidMesh } from '@takram/three-geospatial/r3f'

import localWeatherUrl from '../../../packages/clouds/assets/local_weather.png?url'
import shapeDetailUrl from '../../../packages/clouds/assets/shape_detail.bin?url'
import shapeUrl from '../../../packages/clouds/assets/shape.bin?url'
import turbulenceUrl from '../../../packages/clouds/assets/turbulence.png?url'
import stbnUrl from '../../../packages/core/assets/stbn.bin?url'
import type { StoryFC } from '../components/createStory'
import { Stats } from '../components/Stats'
import {
  localDateArgs,
  localDateArgTypes,
  useLocalDateControls,
  type LocalDateArgs
} from '../controls/localDateControls'
import {
  outputPassArgs,
  outputPassArgTypes,
  type OutputPassArgs
} from '../controls/outputPassControls'
import {
  rendererArgs,
  rendererArgTypes,
  type RendererArgs
} from '../controls/rendererControls'
import {
  toneMappingArgs,
  toneMappingArgTypes,
  type ToneMappingArgs
} from '../controls/toneMappingControls'
import {
  applyCameraMatrixToCamera,
  cameraMatricesApproximatelyEqual,
  readCameraFromUrl,
  readCameraMatrixFromUrl,
  serializeCameraComponents,
  serializeCameraMatrixElements,
  writeCameraToUrl
} from '../helpers/cameraMatrixURL'
import {
  applyPointOfViewToCamera,
  pointOfViewsApproximatelyEqual,
  readPointOfViewFromCamera
} from '../helpers/cameraPointOfView'
import { useControl } from '../hooks/useControl'
import type { PointOfViewProps } from '../hooks/usePointOfView'
import { Story as WebGPUCloudStory } from './Cloud-Story'
import {
  FUJI_PARITY_ANCHOR_LATITUDE,
  FUJI_PARITY_ANCHOR_LONGITUDE,
  FUJI_NO_TILES_LAYER_OPTIONS,
  applyFujiNoTilesCloudParameterPreset
} from './fujiParityPreset'

type Backend = 'webgpu' | 'webgl'
type CloudPresetMode = 'legacy-default' | 'fuji-no-tiles'
type DebugRenderStage = 'final' | 'base' | 'aerial' | 'cloud'

interface FujiParityArgs
  extends LocalDateArgs,
    ToneMappingArgs,
    RendererArgs,
    OutputPassArgs {
  backend: Backend
  googleMapsApiKey: string
  cloudPresetMode: CloudPresetMode
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
  debugDisableOrbitControls?: boolean
  debugDisableCameraFeedback?: boolean
  debugDisableCloudShadowAtlas?: boolean
  debugDisableAerialLighting?: boolean
  debugDisableGeometricCorrection?: boolean
  debugDisableAerialNormal?: boolean
  debugFreezeLocalDate?: boolean
}

const LOCAL_WEATHER_VELOCITY = [0.001, 0] as const
const SHAPE_VELOCITY = [0.00012, 0, 0] as const
const SHAPE_DETAIL_VELOCITY = [0.0015, 0, 0] as const
const WORLD_UP = new Vector3(0, 1, 0)
const targetScratch = new Vector3()
const eyeScratch = new Vector3()
const targetWorldScratch = new Vector3()
const cameraWorldScratch = new Vector3()
const upScratch = new Vector3()
const worldUpScratch = new Vector3()
const pointOfViewQuaternionScratch = new Quaternion()
const matrixScratch = new Matrix4()
const dracoLoader = new DRACOLoader()
dracoLoader.setDecoderPath('https://www.gstatic.com/draco/v1/decoders/')

function getFujiTargetWorld(
  longitude: number,
  latitude: number,
  height = 0
): [number, number, number] {
  const worldToECEFMatrix = new Matrix4()
  const anchorECEF = new Geodetic(
    radians(FUJI_PARITY_ANCHOR_LONGITUDE),
    radians(FUJI_PARITY_ANCHOR_LATITUDE),
    0
  ).toECEF(new Vector3())
  Ellipsoid.WGS84.getNorthUpEastFrame(anchorECEF, worldToECEFMatrix)

  const targetECEF = new Geodetic(
    radians(longitude),
    radians(latitude),
    height
  ).toECEF(new Vector3())

  return targetECEF.applyMatrix4(worldToECEFMatrix.invert()).toArray() as [
    number,
    number,
    number
  ]
}

function getFujiTargetECEF(
  longitude: number,
  latitude: number,
  height = 0
): [number, number, number] {
  return new Geodetic(radians(longitude), radians(latitude), height)
    .toECEF(new Vector3())
    .toArray() as [number, number, number]
}

function mapThreeToneMappingMode(value: ThreeToneMapping): ToneMappingMode {
  switch (value) {
    case LinearToneMapping:
      return ToneMappingMode.LINEAR
    case ReinhardToneMapping:
      return ToneMappingMode.REINHARD
    case CineonToneMapping:
      return ToneMappingMode.CINEON
    case ACESFilmicToneMapping:
      return ToneMappingMode.ACES_FILMIC
    case AgXToneMapping:
    default:
      return ToneMappingMode.AGX
  }
}

function useCloudMotionScale(): number {
  return Math.max(
    useControl(({ cloudMotionScale }: FujiParityArgs) => cloudMotionScale),
    0
  )
}

function useAnimateClouds(): boolean {
  return useControl(({ animateClouds }: FujiParityArgs) => animateClouds)
}

function useCloudPresetMode(): CloudPresetMode {
  return useControl(({ cloudPresetMode }: FujiParityArgs) => cloudPresetMode)
}

function useCloudVelocity2(scale: number): [number, number] {
  const animateClouds = useAnimateClouds()
  return animateClouds
    ? [LOCAL_WEATHER_VELOCITY[0] * scale, LOCAL_WEATHER_VELOCITY[1] * scale]
    : [0, 0]
}

function useCloudVelocity3(
  velocity: readonly [number, number, number],
  scale: number
): [number, number, number] {
  const animateClouds = useAnimateClouds()
  return animateClouds
    ? [velocity[0] * scale, velocity[1] * scale, velocity[2] * scale]
    : [0, 0, 0]
}

const LegacyWebGLGlobe: FC<{ apiKey?: string }> = ({ apiKey }) => (
  <TilesRenderer
    key={apiKey}
    url={`https://tile.googleapis.com/v1/3dtiles/root.json?key=${apiKey}`}
  >
    <TilesPlugin
      plugin={GoogleCloudAuthPlugin}
      args={{
        apiToken: apiKey,
        autoRefreshToken: true
      }}
    />
    <TilesPlugin plugin={GLTFExtensionsPlugin} dracoLoader={dracoLoader} />
    <TilesPlugin plugin={TileCompressionPlugin} />
    <TilesPlugin plugin={UpdateOnChangePlugin} />
    <TilesPlugin plugin={TilesFadePlugin} />
    <TilesAttributionOverlay />
  </TilesRenderer>
)

const StableLegacyGlobeControls: FC = () => {
  const domElement = useThree(({ gl }) => gl.domElement ?? null)
  if (domElement == null) {
    return null
  }

  return (
    <LegacyGlobeControls
      domElement={domElement}
      enableDamping
      adjustHeight={false}
      maxAltitude={Math.PI * 0.55}
    />
  )
}

const LegacyWebGLGlobeAndControls: FC<{ apiKey?: string }> = ({ apiKey }) => {
  const controls = useThree(
    ({ controls }) => controls as GlobeControlsImpl | null
  )

  useLayoutEffect(() => {
    if (controls == null) {
      return
    }
    const callback = (): void => {
      controls.adjustHeight = true
      controls.removeEventListener('start', callback)
    }
    controls.addEventListener('start', callback)
    return () => {
      controls.removeEventListener('start', callback)
    }
  }, [controls])

  return (
    <>
      <LegacyWebGLGlobe apiKey={apiKey} />
      <StableLegacyGlobeControls />
    </>
  )
}

interface FujiCameraSyncProps {
  cameraMatrixElements?: number[] | null
  cameraComponents?: number[] | null
  onCameraMatrixChange?: (elements: number[]) => void
  onCameraComponentsChange?: (components: number[]) => void
}

const LegacyWebGLFujiScene: FC<PointOfViewProps & FujiCameraSyncProps> = ({
  longitude,
  latitude,
  height = 0,
  heading,
  pitch,
  distance,
  cameraMatrixElements = null,
  cameraComponents = null,
  onCameraMatrixChange,
  onCameraComponentsChange
}) => {
  const camera = useThree(({ camera }) => camera)
  const renderer = useThree(({ gl }) => gl)
  const atmosphereRef = useRef<AtmosphereApi>(null)
  const cloudsRef = useRef<ComponentRef<typeof Clouds>>(null)
  const apiKey = useControl(({ googleMapsApiKey }: FujiParityArgs) =>
    googleMapsApiKey !== '' ? googleMapsApiKey : undefined
  )

  const correctAltitude = useControl(
    ({ correctAltitude }: FujiParityArgs) => correctAltitude
  )
  const coverage = useControl(({ coverage }: FujiParityArgs) => coverage)
  const qualityPreset = useControl(
    ({ qualityPreset }: FujiParityArgs) => qualityPreset
  )
  const resolutionScale = useControl(
    ({ resolutionScale }: FujiParityArgs) => resolutionScale
  )
  const temporalUpscale = useControl(
    ({ temporalUpscale }: FujiParityArgs) => temporalUpscale
  )
  const temporalUpscaleScale = useControl(
    ({ temporalUpscaleScale }: FujiParityArgs) => temporalUpscaleScale
  )
  const toneMappingEnabled = useControl(
    ({ toneMapping }: FujiParityArgs) => toneMapping
  )
  const toneMappingMode = useControl(
    ({ toneMappingMode }: FujiParityArgs) => toneMappingMode
  )
  const toneMappingExposure = useControl(
    ({ toneMappingExposure }: FujiParityArgs) => toneMappingExposure
  )
  const shapeDetail = useControl(
    ({ shapeDetail }: FujiParityArgs) => shapeDetail
  )
  const turbulence = useControl(({ turbulence }: FujiParityArgs) => turbulence)
  const haze = useControl(({ haze }: FujiParityArgs) => haze)
  const cloudPresetMode = useCloudPresetMode()
  const applyCloudPresetParameters = useCallback(
    (clouds: ComponentRef<typeof Clouds>): void => {
      if (cloudPresetMode === 'fuji-no-tiles') {
        applyFujiNoTilesCloudParameterPreset(clouds)
      }
    },
    [cloudPresetMode]
  )
  const handleCloudsRef = useCallback(
    (clouds: ComponentRef<typeof Clouds> | null): void => {
      cloudsRef.current = clouds
      if (clouds != null) {
        applyCloudPresetParameters(clouds)
      }
    },
    [applyCloudPresetParameters]
  )

  const cloudMotionScale = useCloudMotionScale()
  const localWeatherVelocity = useCloudVelocity2(cloudMotionScale)
  const shapeVelocity = useCloudVelocity3(SHAPE_VELOCITY, cloudMotionScale)
  const shapeDetailVelocity = useCloudVelocity3(
    SHAPE_DETAIL_VELOCITY,
    cloudMotionScale
  )
  const appliedCameraMatrixSignatureRef = useRef<string>('')
  const emittedCameraMatrixSignatureRef = useRef<string>('')
  const appliedCameraComponentsSignatureRef = useRef<string>('')
  const emittedCameraComponentsSignatureRef = useRef<string>('')
  const fallbackOrbitTarget = useMemo(
    () =>
      new Geodetic(
        radians(cameraComponents?.[0] ?? longitude),
        radians(cameraComponents?.[1] ?? latitude),
        cameraComponents?.[2] ?? height
      )
        .toECEF(new Vector3())
        .toArray() as [number, number, number],
    [cameraComponents, height, latitude, longitude]
  )

  const motionDate = useLocalDateControls(longitude)
  useFrame(() => {
    atmosphereRef.current?.updateByDate(new Date(motionDate.get()))
  })

  useLayoutEffect(() => {
    renderer.toneMapping = NoToneMapping
    renderer.toneMappingExposure = toneMappingExposure
  }, [renderer, toneMappingExposure])

  useLayoutEffect(() => {
    const clouds = cloudsRef.current
    if (clouds == null) {
      return
    }
    applyCloudPresetParameters(clouds)
  }, [applyCloudPresetParameters, qualityPreset])

  useLayoutEffect(() => {
    new PointOfView(distance, radians(heading), radians(pitch)).decompose(
      new Geodetic(radians(longitude), radians(latitude), height).toECEF(
        targetScratch
      ),
      camera.position,
      camera.quaternion,
      camera.up
    )
    camera.updateMatrixWorld()
  }, [camera, distance, heading, height, latitude, longitude, pitch])

  useLayoutEffect(() => {
    if (cameraComponents == null || cameraComponents.length !== 6) {
      return
    }
    const signature = serializeCameraComponents(cameraComponents)
    if (signature === appliedCameraComponentsSignatureRef.current) {
      return
    }
    if (!applyPointOfViewToCamera(camera, cameraComponents)) {
      return
    }
    appliedCameraComponentsSignatureRef.current = signature
    emittedCameraComponentsSignatureRef.current = signature
  }, [camera, cameraComponents])

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
      const components = readPointOfViewFromCamera(camera)
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
  })

  return (
    <Atmosphere ref={atmosphereRef} correctAltitude={correctAltitude}>
      {apiKey != null ? (
        <LegacyWebGLGlobeAndControls apiKey={apiKey} />
      ) : (
        <>
          <EllipsoidMesh args={[Ellipsoid.WGS84.radii, 192, 96]}>
            <meshBasicMaterial color='#6b695e' />
          </EllipsoidMesh>
          <OrbitControls enableDamping target={fallbackOrbitTarget} />
        </>
      )}
      <EffectComposer multisampling={0} enableNormalPass>
        <Fragment
          key={JSON.stringify([
            qualityPreset,
            resolutionScale,
            temporalUpscale,
            temporalUpscaleScale,
            shapeDetail,
            turbulence,
            haze,
            cloudPresetMode,
            toneMappingEnabled,
            toneMappingMode
          ])}
        >
          <Clouds
            ref={handleCloudsRef}
            shadow-farScale={0.25}
            coverage={coverage}
            qualityPreset={qualityPreset}
            resolutionScale={resolutionScale}
            temporalUpscale={temporalUpscale}
            shapeDetail={shapeDetail}
            turbulence={turbulence}
            haze={haze}
            localWeatherVelocity={localWeatherVelocity}
            shapeVelocity={shapeVelocity}
            shapeDetailVelocity={shapeDetailVelocity}
            disableDefaultLayers={cloudPresetMode === 'fuji-no-tiles'}
            localWeatherTexture={localWeatherUrl}
            shapeTexture={shapeUrl}
            shapeDetailTexture={shapeDetailUrl}
            turbulenceTexture={turbulenceUrl}
            stbnTexture={stbnUrl}
          >
            {cloudPresetMode === 'fuji-no-tiles' &&
              FUJI_NO_TILES_LAYER_OPTIONS.map((layer, index) => (
                <CloudLayer
                  key={index}
                  index={index}
                  channel={CloudLayers.DEFAULT[index].channel}
                  shadow={CloudLayers.DEFAULT[index].shadow}
                  {...layer}
                />
              ))}
          </Clouds>
          <AerialPerspective
            sky
            sunLight
            skyLight
            correctGeometricError
            albedoScale={2 / Math.PI}
          />
          {toneMappingEnabled && (
            <>
              <LensFlare />
              <ToneMapping mode={mapThreeToneMappingMode(toneMappingMode)} />
              <SMAA />
              <Dithering />
            </>
          )}
        </Fragment>
      </EffectComposer>
    </Atmosphere>
  )
}

const StableLegacyCanvas: FC<{
  pixelRatio: number
  cameraFar: number
  children: ReactNode
}> = ({ pixelRatio, cameraFar, children }) => {
  const [hostElement, setHostElement] = useState<HTMLDivElement | null>(null)
  const handleHostRef = useCallback((node: HTMLDivElement | null) => {
    setHostElement(current => (current === node ? current : node))
  }, [])

  return (
    <div
      ref={handleHostRef}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%'
      }}
    >
      {hostElement != null && (
        <Canvas
          key='webgl'
          eventSource={hostElement}
          gl={{ depth: false }}
          dpr={pixelRatio}
          camera={{ near: 1, far: cameraFar, up: WORLD_UP.toArray() }}
        >
          <CameraFarSync cameraFar={cameraFar} />
          {children}
        </Canvas>
      )}
    </div>
  )
}

/**
 * Keeps the active R3F camera projection in sync after URL camera changes.
 *
 * @param props Contains the desired far clipping distance.
 * @returns Nothing; updates the active camera projection in-place.
 */
const CameraFarSync: FC<{ cameraFar: number }> = ({ cameraFar }) => {
  const camera = useThree(({ camera }) => camera)

  useLayoutEffect(() => {
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

  return null
}

/**
 * Measures camera distance from URL matrix elements when point-of-view values
 * are not available.
 *
 * @param cameraMatrixElements Matrix-world elements from the URL.
 * @param orbitTarget Current orbit target in the same world frame.
 * @returns Distance in meters when the matrix is usable; otherwise null.
 */
const getCameraMatrixDistance = (
  cameraMatrixElements: number[] | null,
  orbitTarget: [number, number, number]
): number | null => {
  if (cameraMatrixElements == null || cameraMatrixElements.length !== 16) {
    return null
  }
  const distance = Math.hypot(
    cameraMatrixElements[12] - orbitTarget[0],
    cameraMatrixElements[13] - orbitTarget[1],
    cameraMatrixElements[14] - orbitTarget[2]
  )
  return Number.isFinite(distance) && distance > 0 ? distance : null
}

/**
 * Resolves a far clipping distance large enough for low altitude and LEO views.
 *
 * @param cameraComponents Point-of-view URL components when available.
 * @param cameraMatrixElements Matrix-world URL components when available.
 * @param distance Story control fallback camera distance.
 * @param orbitTarget Current orbit target in the active world frame.
 * @returns Far clipping distance shared by WebGL and WebGPU canvases.
 */
const getFujiParityCameraFar = (
  cameraComponents: number[] | null,
  cameraMatrixElements: number[] | null,
  distance: number,
  orbitTarget: [number, number, number]
): number => {
  const cameraDistance =
    cameraComponents?.[5] ??
    getCameraMatrixDistance(cameraMatrixElements, orbitTarget) ??
    distance
  return Math.max(4e5, cameraDistance * 4)
}

const WebGLFujiStory: FC<
  PointOfViewProps & FujiCameraSyncProps & { cameraFar: number }
> = ({ cameraFar, ...props }) => {
  const pixelRatio = useControl(({ pixelRatio }: FujiParityArgs) => pixelRatio)

  return (
    <StableLegacyCanvas pixelRatio={pixelRatio} cameraFar={cameraFar}>
      <LegacyWebGLFujiScene {...props} />
      <Stats />
    </StableLegacyCanvas>
  )
}

/**
 * Fuji parity story that switches between legacy WebGL clouds and the WebGPU
 * no-tiles comparison branch while keeping a single shared control surface.
 *
 * @returns A single-page backend switcher for Fuji cloud comparison.
 */
export const Story: StoryFC<PointOfViewProps, FujiParityArgs> = props => {
  const backend = useControl(({ backend }: FujiParityArgs) => backend)
  const debugDisableOrbitControls = useControl(
    ({ debugDisableOrbitControls = false }: FujiParityArgs) =>
      debugDisableOrbitControls
  )
  const debugDisableCameraFeedback = useControl(
    ({ debugDisableCameraFeedback = false }: FujiParityArgs) =>
      debugDisableCameraFeedback
  )
  const debugDisableCloudShadowAtlas = useControl(
    ({ debugDisableCloudShadowAtlas = false }: FujiParityArgs) =>
      debugDisableCloudShadowAtlas
  )
  const cloudPresetMode = useCloudPresetMode()
  const useFujiNoTilesPreset = cloudPresetMode === 'fuji-no-tiles'
  const useWebglLikeMarchBudget = true
  const useIdentityWorldFrame = cloudPresetMode === 'legacy-default'
  const [cameraComponents, setCameraComponents] = useState<number[] | null>(
    () => readCameraFromUrl()
  )
  const [cameraMatrixElements, setCameraMatrixElements] = useState<
    number[] | null
  >(() => readCameraMatrixFromUrl())
  const [resolvedBackend, setResolvedBackend] = useState<Backend | null>(
    backend
  )
  const pendingBackendRef = useRef<Backend | null>(null)
  const lastCameraMatrixSignatureRef = useRef(
    cameraMatrixElements != null
      ? serializeCameraMatrixElements(cameraMatrixElements)
      : ''
  )
  const lastCameraComponentsSignatureRef = useRef(
    cameraComponents != null ? serializeCameraComponents(cameraComponents) : ''
  )
  const orbitTarget = useMemo(() => {
    const longitude = cameraComponents?.[0] ?? props.longitude
    const latitude = cameraComponents?.[1] ?? props.latitude
    const height = cameraComponents?.[2] ?? props.height ?? 0
    return useIdentityWorldFrame
      ? getFujiTargetECEF(longitude, latitude, height)
      : getFujiTargetWorld(longitude, latitude, height)
  }, [
    cameraComponents,
    props.height,
    props.latitude,
    props.longitude,
    useIdentityWorldFrame
  ])
  const cameraFar = useMemo(
    () =>
      getFujiParityCameraFar(
        cameraComponents,
        cameraMatrixElements,
        props.distance,
        orbitTarget
      ),
    [cameraComponents, cameraMatrixElements, orbitTarget, props.distance]
  )
  const handleCameraComponentsChange = useCallback((components: number[]) => {
    if (components.length !== 6) {
      return
    }
    const signature = serializeCameraComponents(components)
    if (signature === lastCameraComponentsSignatureRef.current) {
      return
    }
    lastCameraComponentsSignatureRef.current = signature
    setCameraComponents(current => {
      if (pointOfViewsApproximatelyEqual(current, components)) {
        return current ?? components
      }
      return components.slice(0, 6)
    })
    writeCameraToUrl(components)
  }, [])

  const handleCameraMatrixChange = useCallback((elements: number[]) => {
    if (elements.length !== 16) {
      return
    }
    const signature = serializeCameraMatrixElements(elements)
    if (signature === lastCameraMatrixSignatureRef.current) {
      return
    }
    lastCameraMatrixSignatureRef.current = signature
    setCameraMatrixElements(current => {
      if (cameraMatricesApproximatelyEqual(current, elements)) {
        return current ?? elements
      }
      return elements.slice(0, 16)
    })
  }, [])

  useLayoutEffect(() => {
    if (debugDisableCameraFeedback || cameraComponents == null) {
      return
    }
    writeCameraToUrl(cameraComponents)
  }, [cameraComponents, debugDisableCameraFeedback, resolvedBackend])

  useLayoutEffect(() => {
    if (backend === resolvedBackend || pendingBackendRef.current === backend) {
      return
    }
    pendingBackendRef.current = backend
    setResolvedBackend(null)
    const frame = requestAnimationFrame(() => {
      pendingBackendRef.current = null
      setResolvedBackend(backend)
    })
    return () => {
      cancelAnimationFrame(frame)
      if (pendingBackendRef.current === backend) {
        pendingBackendRef.current = null
      }
    }
  }, [backend, resolvedBackend])

  if (resolvedBackend == null) {
    return null
  }

  if (resolvedBackend === 'webgl') {
    return (
      <WebGLFujiStory
        key='backend-webgl'
        {...props}
        cameraMatrixElements={cameraMatrixElements}
        cameraComponents={cameraComponents}
        cameraFar={cameraFar}
        onCameraMatrixChange={
          debugDisableCameraFeedback ? undefined : handleCameraMatrixChange
        }
        onCameraComponentsChange={
          debugDisableCameraFeedback ? undefined : handleCameraComponentsChange
        }
      />
    )
  }

  return (
    <WebGPUCloudStory
      key={`backend-webgpu-${cloudPresetMode}`}
      {...props}
      disableTiles
      disableCloudStoryPreset
      useFujiNoTilesCloudPreset={useFujiNoTilesPreset}
      forceWebglLikeMarchBudget={useWebglLikeMarchBudget}
      disableFallbackNoApiKeyCameraOverride
      useIdentityWorldToECEFFrame={useIdentityWorldFrame}
      alignWithWebGLLightingModel
      enableCloudShadowAtlas={
        useFujiNoTilesPreset && !debugDisableCloudShadowAtlas
      }
      updateArgs={() => undefined}
      enableOrbitControls={!debugDisableOrbitControls}
      orbitControlsTarget={orbitTarget}
      cameraFar={cameraFar}
      cameraMatrixElements={cameraMatrixElements}
      cameraComponents={cameraComponents}
      onCameraMatrixChange={
        debugDisableCameraFeedback ? undefined : handleCameraMatrixChange
      }
      onCameraComponentsChange={
        debugDisableCameraFeedback ? undefined : handleCameraComponentsChange
      }
      hideDescription
    />
  )
}

Story.args = {
  backend: 'webgpu',
  googleMapsApiKey: '',
  cloudPresetMode: 'fuji-no-tiles',
  correctAltitude: true,
  coverage: 0.4,
  qualityPreset: 'high',
  resolutionScale: 1,
  taaEnabled: true,
  discardAllHistory: false,
  velocityThresholdPixels: 6,
  historyResetDistanceThreshold: 100,
  temporalAlpha: -1,
  temporalUpscale: false,
  temporalUpscaleScale: 0.375,
  animateClouds: true,
  cloudMotionScale: 1,
  shapeDetail: true,
  turbulence: true,
  haze: true,
  debugRenderStage: 'final',
  debugDisableOrbitControls: false,
  debugDisableCameraFeedback: false,
  debugDisableCloudShadowAtlas: false,
  debugDisableAerialLighting: false,
  debugDisableGeometricCorrection: false,
  debugDisableAerialNormal: false,
  debugFreezeLocalDate: false,
  ...localDateArgs({
    dayOfYear: 200,
    timeOfDay: 17.5
  }),
  ...toneMappingArgs({
    toneMappingMode: AgXToneMapping,
    toneMappingExposure: 10
  }),
  ...outputPassArgs(),
  ...rendererArgs()
}

Story.argTypes = {
  backend: {
    control: {
      type: 'select'
    },
    options: ['webgpu', 'webgl'] satisfies Backend[],
    table: { category: 'parity' }
  },
  googleMapsApiKey: {
    name: 'google maps api key',
    control: 'text',
    table: { category: 'parity' }
  },
  cloudPresetMode: {
    name: 'cloud preset',
    control: {
      type: 'select'
    },
    options: ['legacy-default', 'fuji-no-tiles'] satisfies CloudPresetMode[],
    table: { category: 'parity' }
  },
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
  debugDisableOrbitControls: {
    name: 'disable orbit controls',
    control: {
      type: 'boolean'
    },
    table: { category: 'debug' }
  },
  debugDisableCameraFeedback: {
    name: 'disable camera feedback',
    control: {
      type: 'boolean'
    },
    table: { category: 'debug' }
  },
  debugDisableCloudShadowAtlas: {
    name: 'disable cloud shadow atlas',
    control: {
      type: 'boolean'
    },
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
  ...outputPassArgTypes({
    hasNormal: true,
    hasVelocity: true
  }),
  ...rendererArgTypes()
}

export default Story
