import {
  Matrix4,
  NearestFilter,
  Quaternion,
  Vector2,
  Vector3,
  type Camera,
  type Texture
} from 'three'
import {
  abs,
  Break,
  Continue,
  convertToTexture,
  cross,
  dFdx,
  dFdy,
  dot,
  exp,
  float,
  Fn,
  If,
  ivec2,
  log2,
  Loop,
  max,
  min,
  mix,
  mrt,
  property,
  remapClamp,
  screenCoordinate,
  screenSize,
  select,
  sqrt,
  texture,
  texture3D,
  textureLevel,
  textureSize,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
  viewZToLogarithmicDepth,
  viewZToOrthographicDepth,
  viewZToPerspectiveDepth
} from 'three/tsl'
import {
  NodeMaterial,
  NodeUpdateType,
  QuadMesh,
  RendererUtils,
  TempNode,
  type NodeBuilder,
  type NodeFrame,
  type TextureNode
} from 'three/webgpu'

import {
  getAtmosphereContext,
  getSkyLuminanceToPoint,
  getSunAndSkyIlluminance,
  getSunAndSkyScalarIlluminance
} from '@takram/three-atmosphere/webgpu'
import {
  cameraFar,
  cameraNear,
  cameraPositionWorld,
  depthToViewZ,
  interleavedGradientNoise,
  inverseViewMatrix,
  raySphereIntersection,
  screenToPositionView,
  TemporalAntialiasNode,
  temporalAntialias,
  type Node
} from '@takram/three-geospatial/webgpu'

import {
  CloudsContext,
  fallbackLocalWeatherTexture,
  fallbackShapeDetailTexture,
  fallbackShapeTexture,
  fallbackStbnTexture,
  fallbackTurbulenceTexture,
  getCloudsContext
} from './CloudsContext'
import { CloudsRenderTargets } from './CloudsRenderTargets'
import { CloudsShadowNode } from './CloudsShadowNode'
import { CloudsShadowLengthNode } from './CloudsShadowLengthNode'
import { CloudsTemporalState } from './CloudsTemporalState'
import {
  WEBGPU_MAX_PRIMARY_STEPS,
  WEBGPU_MAX_SECONDARY_STEPS
} from './qualityPresets'

const { resetRendererState, restoreRendererState } = RendererUtils

const MULTI_SCATTERING_OCTAVES = 8
const EPSILON = 1e-6
const WEBGPU_MAX_SHADOW_LENGTH_STEPS = 512
const HIGH_FREQUENCY_FADE_START = 20_000
const HIGH_FREQUENCY_FADE_END = 120_000
const SHAPE_DETAIL_FADE_FLOOR = 0.02
const RECIPROCAL_PI4 = 1 / (4 * Math.PI)
export const CLOUDS_TEMPORAL_ALPHA = 0.05
export const CLOUDS_VARIANCE_GAMMA = 1.0
export const CLOUDS_VELOCITY_THRESHOLD = 0.1
export const CLOUDS_DEPTH_ERROR = 0.001
const CAMERA_CUT_POSITION_THRESHOLD = 1_000
const CAMERA_CUT_ROTATION_THRESHOLD = Math.PI / 12
const CAMERA_CUT_PROJECTION_THRESHOLD = 1e-3
const LOCAL_WEATHER_TANGENT_STEP_WORLD = 10_000
const LOCAL_WEATHER_JACOBIAN_EPSILON = 1e-8
const sizeScratch = /*#__PURE__*/ new Vector2()
const cameraPositionScratch = /*#__PURE__*/ new Vector3()
const cameraQuaternionScratch = /*#__PURE__*/ new Quaternion()

function getMaxMatrixDelta(
  a: Matrix4,
  b: Matrix4,
  { ignoreProjectionJitter = false }: { ignoreProjectionJitter?: boolean } = {}
): number {
  let delta = 0
  for (let i = 0; i < 16; ++i) {
    // Ignore projection center terms (m02/m12 in row-major notation), which
    // are used for temporal jitter and should not trigger camera-cut resets.
    if (ignoreProjectionJitter && (i === 8 || i === 9)) {
      continue
    }
    delta = Math.max(delta, Math.abs(a.elements[i] - b.elements[i]))
  }
  return delta
}

export function updateStbnSamplingParameters(
  texture: Texture,
  frameId: number,
  scale: Vector3,
  layerOverride?: number
): number {
  const image = texture.image as
    | {
        width?: number
        height?: number
        depth?: number
      }
    | undefined
  const width = Math.max(image?.width ?? 1, 1)
  const height = Math.max(image?.height ?? 1, 1)
  const depth = Math.max(image?.depth ?? 1, 1)

  scale.set(1 / width, 1 / height, 1 / depth)
  const layer = layerOverride ?? frameId
  return ((layer % depth) + depth) % depth
}

export function mapScreenCoordinateToLowResCoordinate(
  coord: number,
  fullSize: number,
  lowSize: number
): number {
  if (lowSize <= 1 || fullSize <= 1) {
    return 0
  }
  return Math.min(
    Math.max(Math.floor(((coord + 0.5) * lowSize) / fullSize), 0),
    lowSize - 1
  )
}

const getGlobeUv = /*#__PURE__*/ Fn(([position]: [Node<'vec3'>]) => {
  const n = position.normalize().toConst()
  const f = abs(n).toConst()
  const c = n.div(max(f.x, max(f.y, f.z))).toConst()
  const m = vec2().toVar()

  If(f.y.greaterThan(f.x).and(f.y.greaterThan(f.z)), () => {
    m.assign(select(c.y.greaterThan(0), vec2(n.x.negate(), n.z), n.xz))
  })
    .ElseIf(f.x.greaterThan(f.y).and(f.x.greaterThan(f.z)), () => {
      m.assign(select(c.x.greaterThan(0), n.yz, vec2(n.y.negate(), n.z)))
    })
    .Else(() => {
      m.assign(select(c.z.greaterThan(0), n.xy, vec2(n.x, n.y.negate())))
    })

  const m2 = m.pow2().toConst()
  const q = m2.x.mul(-2).add(m2.y.mul(2)).sub(3).toConst()
  const q2 = q.pow2().toConst()
  const uv = vec2().toVar()

  uv.x.assign(
    sqrt(
      float(1.5)
        .add(m2.x)
        .sub(m2.y)
        .sub(sqrt(m2.x.mul(-24).add(q2)).mul(0.5))
    ).mul(select(m.x.greaterThan(0), 1, -1))
  )
  uv.y.assign(sqrt(float(6).div(float(3).sub(uv.x.pow2()))).mul(m.y))

  return uv.mul(0.5).add(0.5)
})

const dualLobePhase = /*#__PURE__*/ Fn(
  ([cosTheta, g0, g1, mixAmount]: [
    Node<'float'>,
    Node<'float'>,
    Node<'float'>,
    Node<'float'>
  ]) => {
    const g = vec2(g0, g1).toConst()
    const g2 = g.pow2().toConst()
    const lobe = vec2(1)
      .sub(g2)
      .div(
        max(vec2(EPSILON), vec2(1).add(g2).sub(g.mul(cosTheta).mul(2)).pow(1.5))
      )
      .mul(1 / (4 * Math.PI))
      .toConst()
    return lobe.x.mul(mixAmount.oneMinus()).add(lobe.y.mul(mixAmount))
  }
)

const wrapPeriodicDelta = /*#__PURE__*/ Fn(([value]: [Node<'float'>]) =>
  select(
    value.greaterThan(0.5),
    value.sub(1),
    select(value.lessThan(-0.5), value.add(1), value)
  )
)

const wrapPeriodicUvDelta = /*#__PURE__*/ Fn(([value]: [Node<'vec2'>]) =>
  vec2(wrapPeriodicDelta(value.x), wrapPeriodicDelta(value.y))
)

const getBayerIndex = /*#__PURE__*/ Fn(([coord]: [Node<'ivec2'>]) => {
  const x = coord.x.mod(4).toConst()
  const y = coord.y.mod(4).toConst()

  return select(
    x.equal(0),
    select(
      y.equal(0),
      float(0),
      select(y.equal(1), float(12), select(y.equal(2), float(3), float(15)))
    ),
    select(
      x.equal(1),
      select(
        y.equal(0),
        float(8),
        select(y.equal(1), float(4), select(y.equal(2), float(11), float(7)))
      ),
      select(
        x.equal(2),
        select(
          y.equal(0),
          float(2),
          select(y.equal(1), float(14), select(y.equal(2), float(1), float(13)))
        ),
        select(
          y.equal(0),
          float(10),
          select(y.equal(1), float(6), select(y.equal(2), float(9), float(5)))
        )
      )
    )
  )
})

export interface CloudsNodeOptions {
  velocityNode?: TextureNode | null
  normalNode?: Node<'vec3'> | null
}

type CloudsMarchRenderTextureName =
  | 'output'
  | 'effectColor'
  | 'transmittanceDepth'
  | 'velocity'
  | 'cloudDepth'
type CloudsResolvedTextureName = 'output' | 'velocity' | 'mask' | 'depth'
export type CloudsTextureName = Exclude<CloudsMarchRenderTextureName, 'effectColor'>
type CloudsTemporalResolveOwner = {
  projectionMatrix?: Matrix4 | null
}

interface CloudsTemporalResolveOptions {
  temporalAlpha?: number
  varianceGamma?: number
  velocityThreshold?: number
  depthError?: number
  allowBackgroundHistory?: boolean
  disableDepthRejection?: boolean
  disableVelocityRejection?: boolean
}

const resolveNeighborOffsets = [
  /*#__PURE__*/ ivec2(-1, -1),
  /*#__PURE__*/ ivec2(-1, 0),
  /*#__PURE__*/ ivec2(-1, 1),
  /*#__PURE__*/ ivec2(0, -1),
  /*#__PURE__*/ ivec2(0, 0),
  /*#__PURE__*/ ivec2(0, 1),
  /*#__PURE__*/ ivec2(1, -1),
  /*#__PURE__*/ ivec2(1, 0),
  /*#__PURE__*/ ivec2(1, 1)
]

export function createCloudsTemporalResolveNode(
  createTemporalAntialias: typeof temporalAntialias,
  owner: CloudsTemporalResolveOwner,
  outputNode: TextureNode,
  depthNode: TextureNode,
  velocityNode: TextureNode,
  camera: Camera,
  currentFrameMaskNode: TextureNode | null,
  {
    temporalAlpha = CLOUDS_TEMPORAL_ALPHA,
    varianceGamma = CLOUDS_VARIANCE_GAMMA,
    velocityThreshold = CLOUDS_VELOCITY_THRESHOLD,
    depthError = CLOUDS_DEPTH_ERROR,
    allowBackgroundHistory = false,
    disableDepthRejection = false,
    disableVelocityRejection = false
  }: CloudsTemporalResolveOptions = {}
): TemporalAntialiasNode {
  const resolveNode = createTemporalAntialias(owner)(
    outputNode,
    depthNode,
    velocityNode,
    camera,
    currentFrameMaskNode
  )
  resolveNode.temporalAlpha.value = temporalAlpha
  resolveNode.varianceGamma.value = varianceGamma
  resolveNode.velocityThreshold.value = velocityThreshold
  resolveNode.depthError.value = depthError
  resolveNode.allowBackgroundHistory = allowBackgroundHistory
  resolveNode.disableDepthRejection = disableDepthRejection
  resolveNode.disableVelocityRejection = disableVelocityRejection
  return resolveNode
}

export class CloudsNode extends TempNode {
  static override get type(): string {
    return 'CloudsNode'
  }

  inputNode: TextureNode
  depthNode: TextureNode
  velocityNode?: TextureNode | null
  normalNode?: Node<'vec3'> | null
  camera: Camera
  projectionMatrix?: Matrix4 | null

  private readonly marchRenderTargets = new CloudsRenderTargets<CloudsMarchRenderTextureName>(
    this,
    {
      colorAttachments: ['output', 'effectColor', 'transmittanceDepth', 'velocity'],
      depthAttachment: 'cloudDepth'
    }
  )
  private readonly resolvedRenderTargets =
    new CloudsRenderTargets<CloudsResolvedTextureName>(null, {
      colorAttachments: ['output', 'velocity', 'mask'],
      depthAttachment: 'depth'
    })
  private readonly marchMaterial = new NodeMaterial()
  private readonly resolveMaterial = new NodeMaterial()
  private readonly marchMesh = new QuadMesh(this.marchMaterial)
  private readonly resolveMesh = new QuadMesh(this.resolveMaterial)
  private readonly localWeatherNode = texture(fallbackLocalWeatherTexture)
  private readonly shapeTextureNode = texture3D(fallbackShapeTexture)
  private readonly shapeDetailTextureNode = texture3D(fallbackShapeDetailTexture)
  private readonly turbulenceTextureNode = texture(fallbackTurbulenceTexture)
  private readonly stbnTextureNode = texture3D(fallbackStbnTexture)
  private readonly stbnScaleNode = uniform(new Vector3()).setName(
    'cloudsStbnScale'
  )
  private readonly stbnLayerNode = uniform(0).setName('cloudsStbnLayer')
  private readonly resolveFrameIndexNode = uniform(0).setName(
    'cloudsResolveFrameIndex'
  )
  private readonly mipLevelScale = uniform(0.1)
  private readonly currentBaseProjectionMatrix = new Matrix4()
  private readonly previousBaseProjectionMatrix = new Matrix4()
  private readonly currentProjectionMatrixNode = uniform(new Matrix4()).setName(
    'cloudsCurrentProjectionMatrix'
  )
  private readonly previousProjectionMatrixNode = uniform(new Matrix4()).setName(
    'cloudsPreviousProjectionMatrix'
  )
  private readonly currentInverseProjectionMatrixNode = uniform(
    new Matrix4()
  ).setName('cloudsCurrentInverseProjectionMatrix')
  private readonly currentViewMatrixNode = uniform(new Matrix4()).setName(
    'cloudsCurrentViewMatrix'
  )
  private readonly previousViewMatrixNode = uniform(new Matrix4()).setName(
    'cloudsPreviousViewMatrix'
  )
  private readonly resolvedDepthNode = property('float', 'cloudsResolvedDepth')
  private readonly resolveDepthNode = property('float', 'cloudsResolveDepth')

  private rendererState?: RendererUtils.RendererState
  private cloudsContext?: CloudsContext
  private cloudsShadowNode: CloudsShadowNode | null = null
  private cloudsShadowLengthNode: Node<'float'> | null = null
  private readonly temporalState = new CloudsTemporalState()
  private temporalResolveNode?: TemporalAntialiasNode
  private temporalUpscaleEnabled = false
  private temporalFrameIndex = 0
  private animatedStbnFrame = 0
  private previousFrameValid = false
  private cameraPoseValid = false
  private readonly previousCameraPosition = new Vector3()
  private readonly previousCameraQuaternion = new Quaternion()

  constructor(
    inputNode: Node,
    depthNode: TextureNode,
    camera: Camera,
    { velocityNode = null, normalNode = null }: CloudsNodeOptions = {}
  ) {
    super('vec4')

    this.inputNode = convertToTexture(inputNode)
    this.depthNode = depthNode
    this.velocityNode = velocityNode
    this.normalNode = normalNode
    this.camera = camera
    this.marchRenderTargets.getTexture('velocity').minFilter = NearestFilter
    this.marchRenderTargets.getTexture('velocity').magFilter = NearestFilter
    this.marchRenderTargets.getTexture('cloudDepth').minFilter = NearestFilter
    this.marchRenderTargets.getTexture('cloudDepth').magFilter = NearestFilter
    this.updateBeforeType = NodeUpdateType.FRAME
  }

  getTexture(name: CloudsTextureName = 'output'): Texture {
    return this.marchRenderTargets.getTexture(name)
  }

  getTextureNode(name: CloudsTextureName = 'output'): TextureNode {
    return this.marchRenderTargets.getTextureNode(name)
  }

  override updateBefore({ renderer, deltaTime, frameId }: NodeFrame): void {
    if (renderer == null || this.cloudsContext == null) {
      return
    }

    const context = this.cloudsContext
    context.advance(frameId, deltaTime)
    this.localWeatherNode.value = context.resolvedLocalWeatherTexture
    this.shapeTextureNode.value = context.resolvedShapeTexture
    this.shapeDetailTextureNode.value = context.resolvedShapeDetailTexture
    this.turbulenceTextureNode.value = context.resolvedTurbulenceTexture
    this.stbnTextureNode.value = context.resolvedStbnTexture
    this.stbnLayerNode.value = updateStbnSamplingParameters(
      context.resolvedStbnTexture,
      this.animatedStbnFrame,
      this.stbnScaleNode.value,
      context.animateStbn ? undefined : context.stbnFrameIndex
    )
    this.resolveFrameIndexNode.value = this.temporalFrameIndex
    this.temporalState.observe(context)

    const { width, height } = this.getMarchRenderSize(renderer, context)
    const marchResized = this.setRenderTargetSize(
      this.marchRenderTargets,
      width,
      height
    )
    let resolveResized = false
    if (this.temporalUpscaleEnabled) {
      const resolveSize = this.getResolvedRenderSize(renderer)
      resolveResized = this.setRenderTargetSize(
        this.resolvedRenderTargets,
        resolveSize.width,
        resolveSize.height
      )
    }
    const historyResetRequested =
      this.consumeHistoryReset() ||
      marchResized ||
      resolveResized ||
      this.detectCameraCut()
    if (historyResetRequested) {
      this.temporalFrameIndex = 0
    }
    this.updateProjectionMatrices(historyResetRequested, context)
    this.cacheCameraPose()
    if (historyResetRequested) {
      this.temporalResolveNode?.invalidateHistory()
    }

    this.rendererState = resetRendererState(renderer, this.rendererState)
    renderer.setRenderTarget(this.marchRenderTargets.renderTarget)
    this.marchMesh.material = this.marchMaterial
    this.marchMesh.render(renderer)
    if (this.temporalUpscaleEnabled) {
      renderer.setRenderTarget(this.resolvedRenderTargets.renderTarget)
      this.resolveMesh.material = this.resolveMaterial
      this.resolveMesh.render(renderer)
      this.temporalFrameIndex = (this.temporalFrameIndex + 1) % 16
    }
    if (context.animateStbn) {
      this.animatedStbnFrame += 1
    }
    restoreRendererState(renderer, this.rendererState)
  }

  override setup(builder: NodeBuilder): unknown {
    const cloudsContext = getCloudsContext(builder)
    const atmosphereContext = getAtmosphereContext(builder)
    const cloudsShadowNode =
      typeof builder.context.getCloudsShadow === 'function'
        ? builder.context.getCloudsShadow()
        : null
    if (
      cloudsShadowNode != null &&
      !(cloudsShadowNode instanceof CloudsShadowNode)
    ) {
      throw new Error(
        'getCloudsShadow() must return an instanceof CloudsShadowNode.'
      )
    }
    this.cloudsContext = cloudsContext
    this.cloudsShadowNode = cloudsShadowNode
    if (cloudsShadowNode != null) {
      cloudsShadowNode.setContexts(cloudsContext, atmosphereContext)
    }
    const cloudsShadowLengthSource =
      typeof builder.context.getCloudsShadowLength === 'function'
        ? builder.context.getCloudsShadowLength()
        : null
    if (cloudsShadowLengthSource instanceof CloudsShadowLengthNode) {
      cloudsShadowLengthSource.setContexts(cloudsContext, atmosphereContext)
      this.cloudsShadowLengthNode = cloudsShadowLengthSource.sampleShadowLength()
    } else if (
      cloudsShadowLengthSource != null &&
      typeof (
        cloudsShadowLengthSource as {
          sampleShadowLength?: unknown
        }
      ).sampleShadowLength === 'function'
    ) {
      this.cloudsShadowLengthNode = (
        cloudsShadowLengthSource as {
          sampleShadowLength: () => Node<'float'>
        }
      ).sampleShadowLength()
    } else {
      this.cloudsShadowLengthNode = null
    }
    this.temporalUpscaleEnabled = cloudsContext.temporalUpscale
    const temporalAntialiasEnabled = cloudsContext.temporalAntialias

    this.marchMaterial.fragmentNode = this.setupFragmentNode(builder)
    this.marchMaterial.depthNode = this.resolvedDepthNode
    this.marchMaterial.needsUpdate = true

    this.getTextureNode('output').uvNode = this.inputNode.uvNode
    this.getTextureNode('transmittanceDepth').uvNode = this.inputNode.uvNode
    this.getTextureNode('velocity').uvNode = this.inputNode.uvNode
    this.getTextureNode('cloudDepth').uvNode = this.inputNode.uvNode

    if (this.temporalUpscaleEnabled) {
      this.resolveMaterial.fragmentNode = this.setupResolveFragmentNode()
      this.resolveMaterial.depthNode = this.resolveDepthNode
      this.resolveMaterial.needsUpdate = true

      this.resolvedRenderTargets.getTextureNode('output').uvNode =
        this.inputNode.uvNode
      this.resolvedRenderTargets.getTextureNode('velocity').uvNode =
        this.inputNode.uvNode
      this.resolvedRenderTargets.getTextureNode('mask').uvNode =
        this.inputNode.uvNode
      this.resolvedRenderTargets.getTextureNode('depth').uvNode =
        this.inputNode.uvNode

      if (!temporalAntialiasEnabled) {
        this.temporalResolveNode = undefined
        return this.resolvedRenderTargets.getTextureNode('output')
      }

      this.temporalResolveNode = createCloudsTemporalResolveNode(
        temporalAntialias,
        this,
        this.resolvedRenderTargets.getTextureNode('output'),
        this.resolvedRenderTargets.getTextureNode('depth'),
        this.resolvedRenderTargets.getTextureNode('velocity'),
        this.camera,
        null,
        {
          temporalAlpha: 0.03,
          varianceGamma: 4,
          allowBackgroundHistory: true,
          disableDepthRejection: true,
          disableVelocityRejection: true
        }
      )
      return this.temporalResolveNode
    }

    const fullResOutputNode = texture(this.getTexture('output'))
    const fullResDepthNode = texture(this.getTexture('cloudDepth'))
    const fullResVelocityNode = texture(this.getTexture('velocity'))
    fullResOutputNode.uvNode = this.inputNode.uvNode
    fullResDepthNode.uvNode = this.inputNode.uvNode
    fullResVelocityNode.uvNode = this.inputNode.uvNode

    if (!temporalAntialiasEnabled) {
      this.temporalResolveNode = undefined
      return fullResOutputNode
    }

    this.temporalResolveNode = createCloudsTemporalResolveNode(
      temporalAntialias,
      this,
      fullResOutputNode,
      fullResDepthNode,
      fullResVelocityNode,
      this.camera,
      null,
      {
        temporalAlpha: 0.1,
        varianceGamma: 2,
        allowBackgroundHistory: true,
        disableDepthRejection: true,
        disableVelocityRejection: true
      }
    )
    return this.temporalResolveNode
  }

  override dispose(): void {
    this.marchRenderTargets.dispose()
    this.resolvedRenderTargets.dispose()
    this.marchMaterial.dispose()
    this.resolveMaterial.dispose()
    this.marchMesh.geometry.dispose()
    this.resolveMesh.geometry.dispose()
    this.temporalResolveNode?.dispose()
    super.dispose()
  }

  protected consumeHistoryReset(): boolean {
    return this.temporalState.consumeHistoryReset()
  }

  protected get historyResetRequested(): boolean {
    return this.temporalState.historyResetRequested
  }

  private detectCameraCut(): boolean {
    if (!this.cameraPoseValid) {
      return false
    }

    cameraPositionScratch.setFromMatrixPosition(this.camera.matrixWorld)
    cameraQuaternionScratch.setFromRotationMatrix(this.camera.matrixWorld)

    const positionDelta = cameraPositionScratch.distanceTo(
      this.previousCameraPosition
    )
    const rotationDelta =
      2 *
      Math.acos(
        Math.min(
          1,
          Math.abs(cameraQuaternionScratch.dot(this.previousCameraQuaternion))
        )
      )
    const projectionDelta = getMaxMatrixDelta(
      this.projectionMatrix ?? this.camera.projectionMatrix,
      this.currentBaseProjectionMatrix,
      { ignoreProjectionJitter: this.temporalUpscaleEnabled }
    )

    return (
      positionDelta > CAMERA_CUT_POSITION_THRESHOLD ||
      rotationDelta > CAMERA_CUT_ROTATION_THRESHOLD ||
      projectionDelta > CAMERA_CUT_PROJECTION_THRESHOLD
    )
  }

  private cacheCameraPose(): void {
    this.previousCameraPosition.setFromMatrixPosition(this.camera.matrixWorld)
    this.previousCameraQuaternion.setFromRotationMatrix(this.camera.matrixWorld)
    this.cameraPoseValid = true
  }

  private getMarchRenderScale(context: CloudsContext): number {
    return (
      context.resolutionScale *
      (this.temporalUpscaleEnabled ? context.temporalUpscaleScale : 1)
    )
  }

  private getMarchRenderSize(
    renderer: NonNullable<NodeFrame['renderer']>,
    context: CloudsContext
  ): { width: number; height: number } {
    const size = renderer.getDrawingBufferSize(sizeScratch)
    const scale = this.getMarchRenderScale(context)
    return {
      width: Math.max(Math.round(size.x * scale), 1),
      height: Math.max(Math.round(size.y * scale), 1)
    }
  }

  private getResolvedRenderSize(
    renderer: NonNullable<NodeFrame['renderer']>
  ): { width: number; height: number } {
    const size = renderer.getDrawingBufferSize(sizeScratch)
    return {
      width: Math.max(Math.round(size.x), 1),
      height: Math.max(Math.round(size.y), 1)
    }
  }

  private setRenderTargetSize<Name extends string>(
    renderTargets: CloudsRenderTargets<Name>,
    width: number,
    height: number
  ): boolean {
    const { renderTarget } = renderTargets
    if (renderTarget.width === width && renderTarget.height === height) {
      return false
    }
    renderTargets.setSize(width, height)
    return true
  }

  private updateProjectionMatrices(
    resetHistory: boolean,
    context: CloudsContext
  ): void {
    const projectionMatrix = this.projectionMatrix ?? this.camera.projectionMatrix

    if (!this.previousFrameValid || resetHistory) {
      this.previousBaseProjectionMatrix.copy(projectionMatrix)
      this.previousViewMatrixNode.value.copy(this.camera.matrixWorldInverse)
      this.previousFrameValid = true
    } else {
      this.previousBaseProjectionMatrix.copy(this.currentBaseProjectionMatrix)
      this.previousViewMatrixNode.value.copy(this.currentViewMatrixNode.value)
    }

    this.currentBaseProjectionMatrix.copy(projectionMatrix)
    this.currentProjectionMatrixNode.value.copy(this.currentBaseProjectionMatrix)
    this.previousProjectionMatrixNode.value.copy(this.previousBaseProjectionMatrix)

    if (this.temporalUpscaleEnabled) {
      this.mipLevelScale.value = context.temporalUpscaleScale
    } else {
      this.mipLevelScale.value = 1
    }

    this.currentInverseProjectionMatrixNode.value
      .copy(this.currentProjectionMatrixNode.value)
      .invert()
    this.currentViewMatrixNode.value.copy(this.camera.matrixWorldInverse)
  }

  private setupResolveFragmentNode(): Node {
    const fragmentNode = Fn(() => {
      const coord = ivec2(screenCoordinate)
      const effectTextureNode = texture(
        this.marchRenderTargets.getTexture('effectColor')
      )
      const outputTextureNode = texture(
        this.marchRenderTargets.getTexture('output')
      )
      const transmittanceDepthTextureNode = texture(
        this.marchRenderTargets.getTexture('transmittanceDepth')
      )
      const velocityTextureNode = texture(
        this.marchRenderTargets.getTexture('velocity')
      )
      const depthTextureNode = texture(
        this.marchRenderTargets.getTexture('cloudDepth')
      )
      const lowResSize = textureSize(depthTextureNode)
      const sampleUv = vec2(coord).add(0.5).div(screenSize).toConst()
      const lowResCoord = ivec2(
        vec2(coord).add(0.5).mul(vec2(lowResSize)).div(screenSize)
      )
        .clamp(ivec2(0), lowResSize.sub(1))
        .toConst()
      const closestCoord = lowResCoord.toVar()
      const closestDepth = float(1).toVar()
      const farthestDepth = float(0).toVar()

      for (const offset of resolveNeighborOffsets) {
        const neighbor = lowResCoord
          .add(offset)
          .clamp(ivec2(0), lowResSize.sub(1))
          .toConst()
        const neighborDepth = depthTextureNode.load(neighbor).r.toConst()
        If(neighborDepth.lessThan(closestDepth), () => {
          closestCoord.assign(neighbor)
          closestDepth.assign(neighborDepth)
        })
        If(neighborDepth.greaterThan(farthestDepth), () => {
          farthestDepth.assign(neighborDepth)
        })
      }

      this.resolveDepthNode.assign(closestDepth)
      const baseSceneColor = this.inputNode.sample(sampleUv).toConst()
      const reconstructedEffect = effectTextureNode.sample(sampleUv).toConst()
      const reconstructedTransmittanceDepth = transmittanceDepthTextureNode
        .sample(sampleUv)
        .toConst()
      const nearestColor = vec4(
        baseSceneColor.rgb
          .mul(reconstructedTransmittanceDepth.y)
          .add(reconstructedEffect.rgb),
        baseSceneColor.a
      ).toConst()
      const depthRange = farthestDepth.sub(closestDepth).toConst()
      const smoothWeight = float(1)
        .sub(remapClamp(depthRange, float(0.0015), float(0.03)))
        .toConst()
      const reconstructedColor = mix(
        outputTextureNode.load(closestCoord),
        nearestColor,
        smoothWeight
      ).toConst()
      const reconstructedVelocity = velocityTextureNode.load(closestCoord).xyz.toConst()
      const currentFrameMask = getBayerIndex(closestCoord)
        .equal(this.resolveFrameIndexNode)
        .toFloat()

      return mrt({
        output: reconstructedColor,
        velocity: vec4(reconstructedVelocity, 1),
        mask: vec4(currentFrameMask)
      })
    })()

    ;(
      fragmentNode as Node & {
        isOutputStructNode?: boolean
      }
    ).isOutputStructNode = true

    return fragmentNode
  }

  private setupFragmentNode(builder: NodeBuilder): Node {
    const atmosphere = getAtmosphereContext(builder)
    const clouds = getCloudsContext(builder)
    const cloudsShadowNode = this.cloudsShadowNode
    const perspective = this.camera.isPerspectiveCamera
    const logarithmic = builder.renderer.logarithmicDepthBuffer
    const sampleExtinctionAt = Fn(
      ([positionUnit, mipLevel, jitter, highFrequencyWeight]: [
        Node<'vec3'>,
        Node<'float'>,
        Node<'float'>,
        Node<'float'>
      ]) => {
        const unitToWorld = float(1).div(atmosphere.worldToUnit).toConst()
        const height = positionUnit
          .length()
          .sub(atmosphere.bottomRadius)
          .mul(unitToWorld)
          .toConst()
        const extinction = float(0).toVar()

        If(
          height
            .greaterThanEqual(clouds.minHeightNode)
            .and(height.lessThanEqual(clouds.maxHeightNode)),
          () => {
            const weatherUv = getGlobeUv(positionUnit)
            const weatherCoord = weatherUv
              .mul(clouds.localWeatherRepeatNode)
              .add(clouds.localWeatherOffsetNode)
              .toConst()
            const localWeather = textureLevel(
              this.localWeatherNode,
              weatherCoord,
              mipLevel
            ).rgba.toConst()
            const mappedWeather = vec4(
              dot(localWeather, clouds.localWeatherChannelMask0Node),
              dot(localWeather, clouds.localWeatherChannelMask1Node),
              dot(localWeather, clouds.localWeatherChannelMask2Node),
              dot(localWeather, clouds.localWeatherChannelMask3Node)
            )
              .pow(clouds.weatherExponentsNode)
              .toVar()
            const heightFraction = remapClamp(
              vec4(height),
              clouds.minLayerHeightsNode,
              clouds.maxLayerHeightsNode
            ).toConst()
            const biased = heightFraction
              .pow(clouds.shapeAlteringBiasesNode)
              .mul(2)
              .sub(1)
              .clamp(-1, 1)
              .toConst()
            const heightScale = vec4(1).sub(biased.pow2()).toConst()
            const coverageFactor = vec4(1)
              .sub(clouds.coverageNode.mul(heightScale))
              .toConst()
            const weatherDensity = remapClamp(
              mix(mappedWeather, vec4(1), clouds.coverageFilterWidthsNode),
              coverageFactor,
              coverageFactor.add(clouds.coverageFilterWidthsNode)
            ).toVar()
            const weatherDensityMean = weatherDensity.x
              .add(weatherDensity.y)
              .add(weatherDensity.z)
              .add(weatherDensity.w)
              .mul(0.25)
              .toConst()
            const edgeFade = remapClamp(
              weatherDensityMean,
              float(0.05),
              float(0.22)
            ).toConst()
            const positionWorld = positionUnit.mul(unitToWorld).toConst()
            const localWeatherSpeed = clouds.localWeatherOffsetNode
              .length()
              .toConst()
            const evolution = positionWorld
              .normalize()
              .negate()
              .mul(localWeatherSpeed)
              .mul(2e4)
              .toConst()
            const turbulenceAmount = weatherDensity
              .dot(remapClamp(heightFraction, vec4(0.3), vec4(0)))
              .mul(highFrequencyWeight)
              .mul(edgeFade)
              .toConst()
            const turbulence = vec3(0).toVar()
            If(
              clouds.turbulenceNode.and(
                highFrequencyWeight.greaterThan(EPSILON)
              ),
              () => {
                turbulence.assign(
                  this.turbulenceTextureNode
                    .sample(
                      weatherUv
                        .mul(clouds.localWeatherRepeatNode)
                        .mul(clouds.turbulenceRepeatNode)
                    )
                    .rgb.mul(2)
                    .sub(1)
                    .mul(
                      clouds.turbulenceDisplacementNode.mul(turbulenceAmount)
                    )
                )
              }
            )
            const shape = this.shapeTextureNode
              .sample(
                positionWorld
                  .add(evolution)
                  .add(turbulence)
                  .mul(clouds.shapeRepeatNode)
                  .add(clouds.shapeOffsetNode)
              )
              .r.toConst()
            const density = remapClamp(
              weatherDensity,
              vec4(1).sub(shape).mul(clouds.shapeAmountsNode),
              vec4(1)
            ).toVar()
            const densityMean = density.x
              .add(density.y)
              .add(density.z)
              .add(density.w)
              .mul(0.25)
              .toConst()
            const densityEdgeFade = remapClamp(
              densityMean,
              float(0.04),
              float(0.18)
            ).toConst()
            const detailFade = remapClamp(
              highFrequencyWeight
              .mul(densityEdgeFade.pow2())
              .mul(float(1).sub(remapClamp(mipLevel, float(0.1), float(0.85)))),
              float(SHAPE_DETAIL_FADE_FLOOR),
              float(1)
            )
              .toConst()
            const shapeDetailPosition = positionWorld
              .add(turbulence)
              .mul(clouds.shapeDetailRepeatNode)
              .add(clouds.shapeDetailOffsetNode)
              .toConst()
            const detail = textureLevel(
              this.shapeDetailTextureNode,
              shapeDetailPosition,
              mipLevel
            ).r.toConst()
              If(
                clouds.shapeDetailNode.and(detailFade.greaterThan(EPSILON)),
                () => {
                  const modifier = mix(
                    vec4(detail.pow(6)),
                    vec4(1).sub(detail),
                    remapClamp(heightFraction, vec4(0.2), vec4(0.4))
                  )
                  .mul(clouds.shapeDetailAmountsNode)
                  .toConst()
                  const detailedDensity = remapClamp(
                    density.mul(2),
                    modifier.mul(0.5),
                    vec4(1)
                  ).toConst()
                  density.assign(mix(density, detailedDensity, detailFade))
                }
              )
            density.assign(
              density
                .mul(clouds.densityScalesNode)
                .mul(
                  clouds.densityProfileExpTermsNode
                    .mul(
                      exp(
                        clouds.densityProfileExponentsNode.mul(heightFraction)
                      )
                    )
                    .add(
                      clouds.densityProfileLinearTermsNode.mul(heightFraction)
                    )
                    .add(clouds.densityProfileConstantTermsNode)
                )
                .saturate()
            )
            const densitySum = density.x
              .add(density.y)
              .add(density.z)
              .add(density.w)
              .toConst()
            If(densitySum.greaterThan(clouds.minDensityNode), () => {
              extinction.assign(
                densitySum
                  .mul(
                    clouds.scatteringCoefficientNode.add(
                      clouds.absorptionCoefficientNode
                    )
                  )
                  .max(clouds.minExtinctionNode)
              )
            })
          }
        )

        return extinction
      }
    )
    const approximateMultipleScattering = Fn(
      ([opticalDepth, cosTheta]: [Node<'float'>, Node<'float'>]) => {
        const contribution = float(1).toVar()
        const attenuation = float(1).toVar()
        const phaseAttenuation = float(1).toVar()
        const scattering = float(0).toVar()

        Loop(
          { start: 0, end: MULTI_SCATTERING_OCTAVES, condition: '<' },
          () => {
            const beerLambert = exp(
              opticalDepth.negate().mul(attenuation)
            ).toConst()
            scattering.addAssign(
              contribution
                .mul(beerLambert)
                .mul(
                  dualLobePhase(
                    cosTheta,
                    clouds.scatterAnisotropy1Node.mul(phaseAttenuation),
                    clouds.scatterAnisotropy2Node.mul(phaseAttenuation),
                    clouds.scatterAnisotropyMixNode
                  )
                )
            )
            contribution.mulAssign(0.5)
            attenuation.mulAssign(0.5)
            phaseAttenuation.mulAssign(0.5)
          }
        )

        return scattering
      }
    )
    const approximateGroundRadiance = Fn(
      ([positionUnit, surfaceNormal, height, mipLevel, jitter]: [
        Node<'vec3'>,
        Node<'vec3'>,
        Node<'float'>,
        Node<'float'>,
        Node<'float'>
      ]) => {
        const opticalDepthToGround = float(0.5).toVar()

        If(clouds.maxIterationCountToGroundNode.greaterThan(0), () => {
          const secondaryIterationCount = clouds.maxIterationCountToGroundNode
            .toFloat()
            .toConst()
          const secondaryStepSizeWorld = clouds.minSecondaryStepSizeNode
            .div(secondaryIterationCount.max(EPSILON))
            .toVar()
          const secondaryDistanceWorld = secondaryStepSizeWorld
            .mul(jitter)
            .toVar()
          opticalDepthToGround.assign(0)

          Loop(
            { start: 0, end: WEBGPU_MAX_SECONDARY_STEPS, condition: '<' },
            ({ i }) => {
              If(
                float(i).greaterThanEqual(
                  clouds.maxIterationCountToGroundNode.toFloat()
                ),
                () => {
                  Break()
                }
              )

              const secondaryPositionUnit = positionUnit
                .sub(
                  surfaceNormal.mul(
                    secondaryDistanceWorld.mul(atmosphere.worldToUnit)
                  )
                )
                .toConst()
              const secondaryExtinction = sampleExtinctionAt(
                secondaryPositionUnit,
                mipLevel,
                jitter,
                float(1)
              ).toConst()

              opticalDepthToGround.addAssign(
                secondaryExtinction.mul(secondaryStepSizeWorld)
              )
              secondaryDistanceWorld.addAssign(secondaryStepSizeWorld)
              secondaryStepSizeWorld.mulAssign(clouds.secondaryStepScaleNode)
            }
          )
        })

        const groundPointUnit = positionUnit
          .sub(surfaceNormal.mul(height.mul(atmosphere.worldToUnit)))
          .toConst()
        const groundIlluminance = getSunAndSkyIlluminance(
          groundPointUnit,
          surfaceNormal,
          atmosphere.sunDirectionECEF
        ).toConst()
        const groundIrradiance = groundIlluminance
          .get('skyIlluminance')
          .add(
            groundIlluminance
              .get('sunIlluminance')
              .mul(clouds.coverageNode.oneMinus())
          )
          .toConst()
        const bouncedRadiance = groundIrradiance.mul(0.3 / Math.PI).toConst()

        return bouncedRadiance.mul(exp(opticalDepthToGround.negate()))
      }
    )
    const approximateHaze = Fn(
      ([
        rayOriginUnit,
        rayDirectionUnit,
        maxRayDistanceWorld,
        cosTheta,
        shadowLengthWorld
      ]: [
        Node<'vec3'>,
        Node<'vec3'>,
        Node<'float'>,
        Node<'float'>,
        Node<'float'>
      ]) => {
        const haze = vec4(0).toVar()
        const modulation = remapClamp(
          clouds.coverageNode,
          float(0.2),
          float(0.4)
        ).toConst()

        If(
          clouds.hazeNode.and(
            atmosphere.cameraHeight.mul(modulation).lessThan(float(0)).not()
          ),
          () => {
            const density = modulation
              .mul(clouds.hazeDensityScaleNode)
              .mul(exp(atmosphere.cameraHeight.negate().mul(clouds.hazeExponentNode)))
              .toConst()

            If(density.lessThan(float(1e-7)).not(), () => {
              const normalAtOrigin = rayOriginUnit.normalize().toConst()
              const normalAtHorizon = rayOriginUnit
                .sub(rayDirectionUnit.mul(rayOriginUnit.dot(rayDirectionUnit)))
                .div(atmosphere.bottomRadius)
                .toConst()
              const horizonBlend = remapClamp(
                normalAtOrigin.dot(normalAtHorizon),
                float(0.9),
                float(1)
              ).toConst()
              const normal = mix(
                normalAtOrigin,
                normalAtHorizon,
                horizonBlend
              ).toConst()
              const angle = max(normal.dot(rayDirectionUnit), float(1e-5)).toConst()
              const exponent = angle.mul(clouds.hazeExponentNode).toConst()
              const linearTerm = density
                .div(clouds.hazeExponentNode.max(EPSILON))
                .div(angle)
                .toConst()
              const expTerm = float(1)
                .sub(exp(maxRayDistanceWorld.negate().mul(exponent)))
                .toConst()
              const shadowExpTerm = float(1)
                .sub(
                  exp(
                    min(maxRayDistanceWorld, shadowLengthWorld)
                      .negate()
                      .mul(exponent)
                  )
                )
                .toConst()
              const opticalDepth = expTerm.mul(linearTerm).toConst()
              const shadowOpticalDepth = expTerm
                .sub(shadowExpTerm)
                .mul(linearTerm)
                .max(float(0))
                .toConst()
              const transmittance = float(1)
                .sub(exp(opticalDepth.negate()))
                .saturate()
                .toConst()
              const shadowTransmittance = float(1)
                .sub(exp(shadowOpticalDepth.negate()))
                .saturate()
                .toConst()
              const illumination = getSunAndSkyScalarIlluminance(
                rayOriginUnit,
                atmosphere.sunDirectionECEF
              ).toConst()
              const inscatter = illumination
                .get('sunIlluminance')
                .mul(
                  dualLobePhase(
                    cosTheta,
                    clouds.scatterAnisotropy1Node,
                    clouds.scatterAnisotropy2Node,
                    clouds.scatterAnisotropyMixNode
                  )
                )
                .mul(shadowTransmittance)
                .add(
                  illumination
                    .get('skyIlluminance')
                    .mul(float(RECIPROCAL_PI4))
                    .mul(clouds.skyLightScaleNode)
                    .mul(transmittance)
                )
                .mul(
                  clouds.hazeScatteringCoefficientNode.div(
                    clouds.hazeAbsorptionCoefficientNode.add(
                      clouds.hazeScatteringCoefficientNode
                    )
                  )
                )
                .toConst()

              haze.assign(vec4(inscatter, transmittance))
            })
          }
        )

        return haze
      }
    )
    const marchShadowLengthWorld =
      cloudsShadowNode == null || !clouds.lightShafts
        ? null
        : Fn(
            ([
              rayOriginWorld,
              rayDirectionWorld,
              maxRayDistanceWorld,
              jitter
            ]: [
              Node<'vec3'>,
              Node<'vec3'>,
              Node<'float'>,
              Node<'float'>
            ]) => {
              const shadowLengthWorld = float(0).toVar()

              If(maxRayDistanceWorld.greaterThan(EPSILON), () => {
                const stepSizeWorld = clouds.minShadowLengthStepSizeNode.toVar()
                const rayDistanceWorld = stepSizeWorld.mul(jitter).toVar()

                Loop(
                  {
                    start: 0,
                    end: WEBGPU_MAX_SHADOW_LENGTH_STEPS,
                    condition: '<'
                  },
                  ({ i }) => {
                    If(
                      float(i).greaterThanEqual(
                        clouds.maxShadowLengthIterationCountNode.toFloat()
                      ),
                      () => {
                        Break()
                      }
                    )
                    If(rayDistanceWorld.greaterThan(maxRayDistanceWorld), () => {
                      Break()
                    })

                    const samplePositionWorld = rayOriginWorld
                      .add(rayDirectionWorld.mul(rayDistanceWorld))
                      .toConst()
                    const opticalDepth = cloudsShadowNode
                      .sampleOpticalDepth(samplePositionWorld, float(0), null, {
                        cascadeJitterNode: jitter
                      })
                      .toConst()
                    shadowLengthWorld.addAssign(
                      float(1)
                        .sub(exp(opticalDepth.negate()))
                        .mul(stepSizeWorld)
                    )
                    stepSizeWorld.mulAssign(clouds.perspectiveStepScaleNode)
                    rayDistanceWorld.addAssign(stepSizeWorld)
                  }
                )
              })

              return shadowLengthWorld
            }
          )

    const fragmentNode = Fn(() => {
      const texCoord = uv()
      const depthTexCoord = texCoord.toConst()
      const stbnCoord = vec3(screenCoordinate.xy, this.stbnLayerNode)
        .mul(this.stbnScaleNode)
        .toConst()
      const stbnJitter = this.stbnTextureNode.sample(stbnCoord).r.toConst()
      const ignJitter = interleavedGradientNoise(
        vec2(screenCoordinate.xy)
      ).toConst()
      const stochasticJitter = select(clouds.useStbnNode, stbnJitter, ignJitter)
        .clamp(0.01, 0.99)
        .toConst()
      const sceneColor = this.inputNode.sample(texCoord).toVar()
      const effectColor = vec4(0, 0, 0, 1).toVar()
      const transmittanceDepth = vec4(0, 1, 0, 0).toVar()
      const cloudVelocity = vec4(0).toVar()
      sceneColor.a.assign(1)
      const baseSceneColor = sceneColor.rgb.toConst()
      const depth = this.depthNode.sample(depthTexCoord).r.toConst()
      this.resolvedDepthNode.assign(depth)

      const near = cameraNear(this.camera)
      const far = cameraFar(this.camera)
      const currentProjection = this.currentProjectionMatrixNode
      const currentInverseProjection = this.currentInverseProjectionMatrixNode
      const previousProjection = this.previousProjectionMatrixNode
      const currentView = this.currentViewMatrixNode
      const previousView = this.previousViewMatrixNode
      const matrixWorld = inverseViewMatrix(this.camera)
      const cameraWorld = cameraPositionWorld(this.camera).toConst()
      const encodeDepth = (viewZ: Node<'float'>): Node<'float'> =>
        logarithmic
          ? viewZToLogarithmicDepth(viewZ, near, far)
          : perspective
            ? viewZToPerspectiveDepth(viewZ, near, far)
            : viewZToOrthographicDepth(viewZ, near, far)

      const farViewZ = depthToViewZ(float(1), near, far, {
        perspective,
        logarithmic
      }).toConst()
      const farPositionView = screenToPositionView(
        texCoord,
        float(1),
        farViewZ,
        currentProjection,
        currentInverseProjection
      ).toConst()
      const farDirectionWorld = matrixWorld
        .mul(vec4(farPositionView.normalize(), 0))
        .xyz.normalize()
        .toConst()

      const sceneDepthBlend = float(1)
        .sub(remapClamp(depth, float(0.999), float(1)))
        .toConst()
      const sceneViewZ = depthToViewZ(depth, near, far, {
        perspective,
        logarithmic
      }).toConst()
      const scenePositionView = screenToPositionView(
        texCoord,
        depth,
        sceneViewZ,
        currentProjection,
        currentInverseProjection
      ).toConst()
      const scenePositionWorld = matrixWorld
        .mul(vec4(scenePositionView, 1))
        .xyz.toConst()
      const rayDirectionWorld = mix(
        farDirectionWorld,
        scenePositionWorld.sub(cameraWorld).normalize(),
        sceneDepthBlend
      ).normalize()

      const rayDirectionUnit = atmosphere.matrixWorldToECEF
        .mul(vec4(rayDirectionWorld, 0))
        .xyz.normalize()
        .toConst()
      const altitudeCorrectionUnit = select(
        clouds.correctAltitudeNode,
        atmosphere.altitudeCorrectionUnit,
        vec3(0)
      ).toConst()
      const unitToWorld = float(1).div(atmosphere.worldToUnit).toConst()
      const safeShapeRepeat = max(clouds.shapeRepeatNode.abs(), vec3(EPSILON))
        .toConst()
      const safeShapeDetailRepeat = max(
        clouds.shapeDetailRepeatNode.abs(),
        vec3(EPSILON)
      ).toConst()
      const safeLocalWeatherRepeat = max(
        clouds.localWeatherRepeatNode.abs(),
        vec2(EPSILON)
      ).toConst()
      const localWeatherOffsetDeltaUv = wrapPeriodicUvDelta(
        clouds.localWeatherOffsetNode
          .sub(clouds.previousLocalWeatherOffsetNode)
          .div(safeLocalWeatherRepeat)
      ).toConst()
      const shapeMotionWorld = clouds.shapeOffsetNode
        .sub(clouds.previousShapeOffsetNode)
        .div(safeShapeRepeat)
        .toConst()
      const shapeDetailMotionWorld = clouds.shapeDetailOffsetNode
        .sub(clouds.previousShapeDetailOffsetNode)
        .div(safeShapeDetailRepeat)
        .toConst()
      const currentLocalWeatherSpeed = clouds.localWeatherOffsetNode
        .length()
        .toConst()
      const previousLocalWeatherSpeed = clouds.previousLocalWeatherOffsetNode
        .length()
        .toConst()
      const approximateLocalWeatherMotionWorld = Fn(
        ([positionUnit]: [Node<'vec3'>]) => {
          const localWeatherMotionWorld = vec3(0).toVar()

          If(localWeatherOffsetDeltaUv.length().greaterThan(EPSILON), () => {
            const surfaceNormalUnit = positionUnit.normalize().toConst()
            const tangentReferenceUnit = select(
              surfaceNormalUnit.y.abs().greaterThan(0.95),
              vec3(1, 0, 0),
              vec3(0, 1, 0)
            ).toConst()
            const tangentEastUnit = cross(
              tangentReferenceUnit,
              surfaceNormalUnit
            )
              .normalize()
              .toConst()
            const tangentNorthUnit = cross(
              surfaceNormalUnit,
              tangentEastUnit
            )
              .normalize()
              .toConst()
            const tangentStepUnit = float(LOCAL_WEATHER_TANGENT_STEP_WORLD)
              .mul(atmosphere.worldToUnit)
              .toConst()
            const baseUv = getGlobeUv(positionUnit).toConst()
            const eastUvDelta = wrapPeriodicUvDelta(
              getGlobeUv(positionUnit.add(tangentEastUnit.mul(tangentStepUnit)))
                .sub(baseUv)
            ).toConst()
            const northUvDelta = wrapPeriodicUvDelta(
              getGlobeUv(positionUnit.add(tangentNorthUnit.mul(tangentStepUnit)))
                .sub(baseUv)
            ).toConst()
            const determinant = eastUvDelta.x
              .mul(northUvDelta.y)
              .sub(eastUvDelta.y.mul(northUvDelta.x))
              .toConst()

            If(
              abs(determinant).greaterThan(LOCAL_WEATHER_JACOBIAN_EPSILON),
              () => {
                const eastWeight = localWeatherOffsetDeltaUv.x
                  .mul(northUvDelta.y)
                  .sub(localWeatherOffsetDeltaUv.y.mul(northUvDelta.x))
                  .div(determinant)
                  .toConst()
                const northWeight = eastUvDelta.x
                  .mul(localWeatherOffsetDeltaUv.y)
                  .sub(eastUvDelta.y.mul(localWeatherOffsetDeltaUv.x))
                  .div(determinant)
                  .toConst()
                const localWeatherMotionECEF = tangentEastUnit
                  .mul(eastWeight)
                  .add(tangentNorthUnit.mul(northWeight))
                  .mul(tangentStepUnit)
                  .mul(unitToWorld)
                  .toConst()

                localWeatherMotionWorld.assign(
                  atmosphere.matrixECEFToWorld
                    .mul(vec4(localWeatherMotionECEF, 0))
                    .xyz
                )
              }
            )
          })

          return localWeatherMotionWorld
        }
      )
      const cameraPositionUnit = atmosphere.cameraPositionUnit
        .add(altitudeCorrectionUnit)
        .toConst()
      const scenePositionUnit = atmosphere.matrixWorldToECEF
        .mul(vec4(scenePositionWorld, 1))
        .xyz.mul(atmosphere.worldToUnit)
        .add(altitudeCorrectionUnit)
        .toConst()

      // Match legacy semantics: clamp against scene hit distance along the
      // current view ray, not the Euclidean point-to-point distance. Blend the
      // clamp near depth==1 to avoid horizon discontinuities.
      const sceneDistance = mix(
        clouds.maxRayDistanceNode.mul(atmosphere.worldToUnit),
        max(scenePositionUnit.sub(cameraPositionUnit).dot(rayDirectionUnit), 0),
        sceneDepthBlend
      ).toConst()
      const sceneDistanceWorld = sceneDistance.mul(unitToWorld).toConst()
      const globalShadowLengthWorld = (this.cloudsShadowLengthNode ?? float(0))
        .mul(unitToWorld)
        .toConst()
      const shadowLengthWorld = globalShadowLengthWorld.toVar()
      const sunCosTheta = rayDirectionUnit
        .negate()
        .dot(atmosphere.sunDirectionECEF)
        .toConst()

      const maxRadius = atmosphere.bottomRadius
        .add(clouds.maxHeightNode.mul(atmosphere.worldToUnit))
        .toConst()
      const minRadius = atmosphere.bottomRadius
        .add(clouds.minHeightNode.mul(atmosphere.worldToUnit))
        .toConst()
      const shadowTopRadius = atmosphere.bottomRadius
        .add(clouds.shadowMaxHeightNode.mul(atmosphere.worldToUnit))
        .toConst()
      const groundIntersections = raySphereIntersection(
        cameraPositionUnit,
        rayDirectionUnit,
        vec3(0),
        atmosphere.bottomRadius
      ).toConst()
      const maxIntersections = raySphereIntersection(
        cameraPositionUnit,
        rayDirectionUnit,
        vec3(0),
        maxRadius
      ).toConst()
      const minIntersections = raySphereIntersection(
        cameraPositionUnit,
        rayDirectionUnit,
        vec3(0),
        minRadius
      ).toConst()
      const shadowTopIntersections = raySphereIntersection(
        cameraPositionUnit,
        rayDirectionUnit,
        vec3(0),
        shadowTopRadius
      ).toConst()
      const groundHit = cameraPositionUnit
        .dot(rayDirectionUnit)
        .lessThan(0)
        .and(groundIntersections.y.greaterThan(EPSILON))
        .toConst()
      const hazeNearFarWorld = vec2(near, near).toVar()
      const shadowNearFarWorld = vec2(-1).toVar()

      If(atmosphere.cameraHeight.lessThan(clouds.maxHeightNode), () => {
        hazeNearFarWorld.assign(
          vec2(
            near,
            select(groundHit, groundIntersections.x, maxIntersections.y).mul(
              unitToWorld
            )
          )
        )
      }).Else(() => {
        hazeNearFarWorld.assign(vec2(near, maxIntersections.y.mul(unitToWorld)))
        If(groundHit, () => {
          hazeNearFarWorld.y.assign(groundIntersections.x.mul(unitToWorld))
        })
      })

      hazeNearFarWorld.y.assign(min(hazeNearFarWorld.y, sceneDistanceWorld))

      If(
        clouds.lightShaftsNode.and(
          clouds.shadowMaxHeightNode.greaterThan(clouds.shadowMinHeightNode)
        ),
        () => {
          If(
            atmosphere.cameraHeight.lessThan(clouds.shadowMaxHeightNode),
            () => {
              shadowNearFarWorld.assign(
                vec2(
                  near,
                  select(
                    groundHit,
                    groundIntersections.x,
                    shadowTopIntersections.y
                  ).mul(unitToWorld)
                )
              )
            }
          ).Else(() => {
            shadowNearFarWorld.assign(
              vec2(
                max(shadowTopIntersections.x, 0).mul(unitToWorld),
                select(
                  groundHit,
                  min(shadowTopIntersections.y, groundIntersections.x),
                  shadowTopIntersections.y
                ).mul(unitToWorld)
              )
            )
          })
          shadowNearFarWorld.y.assign(min(shadowNearFarWorld.y, sceneDistanceWorld))
        }
      )

      const segmentStart = float(-1).toVar()
      const segmentEnd = float(-1).toVar()

      If(atmosphere.cameraHeight.lessThan(clouds.minHeightNode), () => {
        If(groundHit.not(), () => {
          segmentStart.assign(max(minIntersections.y, 0))
          segmentEnd.assign(
            min(
              maxIntersections.y,
              clouds.maxRayDistanceNode.mul(atmosphere.worldToUnit)
            )
          )
        })
      })
        .ElseIf(atmosphere.cameraHeight.lessThan(clouds.maxHeightNode), () => {
          segmentStart.assign(0)
          segmentEnd.assign(
            select(groundHit, groundIntersections.x, maxIntersections.y)
          )
        })
        .Else(() => {
          segmentStart.assign(max(maxIntersections.x, 0))
          segmentEnd.assign(
            select(
              groundHit,
              min(maxIntersections.y, minIntersections.x),
              maxIntersections.y
            )
          )
        })

      segmentEnd.assign(min(segmentEnd, sceneDistance))

      If(segmentEnd.greaterThan(segmentStart), () => {
        const transmittance = float(1).toVar()
        const radiance = vec3(0).toVar()
        const transmittanceWeightedDepth = float(0).toVar()
        const transmittanceWeightSum = float(0).toVar()
        const rayOriginUnit = cameraPositionUnit
          .add(rayDirectionUnit.mul(segmentStart))
          .toConst()
        const rayOriginWeatherCoord = getGlobeUv(rayOriginUnit)
          .mul(clouds.localWeatherRepeatNode)
          .toConst()
        const derivativeCoord = rayOriginWeatherCoord.mul(screenSize).toConst()
        const derivativeX = dFdx(derivativeCoord).toConst()
        const derivativeY = dFdy(derivativeCoord).toConst()
        const deltaMaxSqr = max(
          derivativeX.dot(derivativeX),
          derivativeY.dot(derivativeY)
        )
          .mul(0.1)
          .toConst()
        const rayStartMipLevel = max(
          float(0),
          log2(max(float(1), deltaMaxSqr)).mul(0.5)
        )
          .mul(this.mipLevelScale)
          .toConst()
        const altitudeMipBlend = min(
          float(1),
          atmosphere.cameraHeight
            .mul(0.2)
            .div(clouds.maxHeightNode.max(EPSILON))
        ).toConst()
        const rayStartTexelsPerPixel = float(2)
          .pow(mix(float(0), rayStartMipLevel, altitudeMipBlend))
          .toConst()
        const segmentStartWorld = segmentStart.mul(unitToWorld).toConst()
        const stepSizeWorld = clouds.minStepSizeNode
          .add(clouds.perspectiveStepScaleNode.sub(1).mul(segmentStartWorld))
          .toVar()
        const rayDistance = segmentStart
          .add(
            stepSizeWorld
              .mul(atmosphere.worldToUnit)
              .mul(stochasticJitter)
              .mul(2)
          )
          .toVar()
        Loop(
          { start: 0, end: WEBGPU_MAX_PRIMARY_STEPS, condition: '<' },
          ({ i }) => {
            If(float(i).greaterThanEqual(clouds.maxIterationCountNode), () => {
              Break()
            })
            If(rayDistance.greaterThanEqual(segmentEnd), () => {
              Break()
            })
            If(transmittance.lessThanEqual(clouds.minTransmittanceNode), () => {
              Break()
            })

            const progress = rayDistance
              .sub(segmentStart)
              .div(segmentEnd.sub(segmentStart).max(EPSILON))
              .saturate()
              .toConst()
            const rayStepSizeWorld = stepSizeWorld.toConst()
            const positionUnit = cameraPositionUnit
              .add(rayDirectionUnit.mul(rayDistance))
              .toConst()
            const rayDistanceWorld = rayDistance.mul(unitToWorld).toConst()
            const mipLevel = log2(
              max(
                float(1),
                rayStartTexelsPerPixel.add(rayDistanceWorld.mul(1e-5))
              )
            ).toConst()
            const height = positionUnit
              .length()
              .sub(atmosphere.bottomRadius)
              .mul(unitToWorld)
              .toConst()

            If(
              height
                .lessThan(clouds.minHeightNode)
                .or(height.greaterThan(clouds.maxHeightNode)),
              () => {
                stepSizeWorld.mulAssign(clouds.perspectiveStepScaleNode)
                rayDistance.addAssign(
                  mix(
                    stepSizeWorld,
                    clouds.maxStepSizeNode,
                    min(float(1), mipLevel)
                  ).mul(atmosphere.worldToUnit)
                )
                Continue()
              }
            )

            const weatherUv = getGlobeUv(positionUnit)
            const weatherCoord = weatherUv
              .mul(clouds.localWeatherRepeatNode)
              .add(clouds.localWeatherOffsetNode)
              .toConst()
            const localWeather = textureLevel(
              this.localWeatherNode,
              weatherCoord,
              mipLevel
            ).rgba.toConst()
            const mappedWeather = vec4(
              dot(localWeather, clouds.localWeatherChannelMask0Node),
              dot(localWeather, clouds.localWeatherChannelMask1Node),
              dot(localWeather, clouds.localWeatherChannelMask2Node),
              dot(localWeather, clouds.localWeatherChannelMask3Node)
            )
              .pow(clouds.weatherExponentsNode)
              .toVar()

            const heightFraction = remapClamp(
              vec4(height),
              clouds.minLayerHeightsNode,
              clouds.maxLayerHeightsNode
            ).toConst()
            const biased = heightFraction
              .pow(clouds.shapeAlteringBiasesNode)
              .mul(2)
              .sub(1)
              .clamp(-1, 1)
              .toConst()
            const heightScale = vec4(1).sub(biased.pow2()).toConst()
            const coverageFactor = vec4(1)
              .sub(clouds.coverageNode.mul(heightScale))
              .toConst()
            const weatherDensity = remapClamp(
              mix(mappedWeather, vec4(1), clouds.coverageFilterWidthsNode),
              coverageFactor,
              coverageFactor.add(clouds.coverageFilterWidthsNode)
            ).toVar()
            const weatherDensityMean = weatherDensity.x
              .add(weatherDensity.y)
              .add(weatherDensity.z)
              .add(weatherDensity.w)
              .mul(0.25)
              .toConst()
            const hasRoughDensity = weatherDensity
              .greaterThan(vec4(clouds.minDensityNode))
              .any()
              .toConst()
            const edgeFade = remapClamp(
              weatherDensityMean,
              float(0.05),
              float(0.22)
            ).toConst()

            If(hasRoughDensity.not(), () => {
              stepSizeWorld.mulAssign(clouds.perspectiveStepScaleNode)
              rayDistance.addAssign(
                mix(
                  stepSizeWorld,
                  clouds.maxStepSizeNode,
                  min(float(1), mipLevel)
                ).mul(atmosphere.worldToUnit)
              )
              Continue()
            })

            const positionWorld = positionUnit.mul(unitToWorld).toConst()
            const localWeatherSpeed = clouds.localWeatherOffsetNode
              .length()
              .toConst()
            const evolution = positionWorld
              .normalize()
              .negate()
              .mul(localWeatherSpeed)
              .mul(2e4)
              .toConst()
            const highFrequencyWeight = float(1)
              .sub(
                remapClamp(
                  rayDistanceWorld,
                  float(HIGH_FREQUENCY_FADE_START),
                  float(HIGH_FREQUENCY_FADE_END)
                )
              )
              .toConst()
            const turbulenceAmount = weatherDensity
              .dot(remapClamp(heightFraction, vec4(0.3), vec4(0)))
              .mul(highFrequencyWeight)
              .mul(edgeFade)
              .toConst()
            const turbulence = vec3(0).toVar()
            If(
              clouds.turbulenceNode.and(
                highFrequencyWeight.greaterThan(EPSILON)
              ),
              () => {
                turbulence.assign(
                  textureLevel(
                    this.turbulenceTextureNode,
                    weatherUv
                      .mul(clouds.localWeatherRepeatNode)
                      .mul(clouds.turbulenceRepeatNode),
                    mipLevel
                  )
                    .rgb.mul(2)
                    .sub(1)
                    .mul(
                      clouds.turbulenceDisplacementNode.mul(turbulenceAmount)
                    )
                )
              }
            )

            const shape = textureLevel(
              this.shapeTextureNode,
              positionWorld
                .add(evolution)
                .add(turbulence)
                .mul(clouds.shapeRepeatNode)
                .add(clouds.shapeOffsetNode),
              mipLevel
            )
              .r.toConst()
            let density = remapClamp(
              weatherDensity,
              vec4(1).sub(shape).mul(clouds.shapeAmountsNode),
              vec4(1)
            ).toVar()
            const densityMean = density.x
              .add(density.y)
              .add(density.z)
              .add(density.w)
              .mul(0.25)
              .toConst()
            const densityEdgeFade = remapClamp(
              densityMean,
              float(0.04),
              float(0.18)
            ).toConst()
            const detailFade = remapClamp(
              highFrequencyWeight
              .mul(densityEdgeFade.pow2())
              .mul(float(1).sub(remapClamp(mipLevel, float(0.1), float(0.85)))),
              float(SHAPE_DETAIL_FADE_FLOOR),
              float(1)
            )
              .toConst()
            const shapeDetailPosition = positionWorld
              .add(turbulence)
              .mul(clouds.shapeDetailRepeatNode)
              .add(clouds.shapeDetailOffsetNode)
              .toConst()
            const detail = textureLevel(
              this.shapeDetailTextureNode,
              shapeDetailPosition,
              mipLevel
            ).r.toConst()
              If(
                clouds.shapeDetailNode.and(detailFade.greaterThan(EPSILON)),
                () => {
                  const modifier = mix(
                    vec4(detail.pow(6)),
                    vec4(1).sub(detail),
                    remapClamp(heightFraction, vec4(0.2), vec4(0.4))
                  )
                  .mul(clouds.shapeDetailAmountsNode)
                  .toConst()
                  const detailedDensity = remapClamp(
                    density.mul(2),
                    modifier.mul(0.5),
                    vec4(1)
                  ).toConst()
                  density.assign(mix(density, detailedDensity, detailFade))
                }
              )

            density.assign(
              density
                .mul(clouds.densityScalesNode)
                .mul(
                  clouds.densityProfileExpTermsNode
                    .mul(
                      exp(
                        clouds.densityProfileExponentsNode.mul(heightFraction)
                      )
                    )
                    .add(
                      clouds.densityProfileLinearTermsNode.mul(heightFraction)
                    )
                    .add(clouds.densityProfileConstantTermsNode)
                )
                .saturate()
            )

            const densitySum = density.x
              .add(density.y)
              .add(density.z)
              .add(density.w)
              .toConst()

            const correctedPosition = positionUnit.toConst()

            If(densitySum.greaterThan(clouds.minDensityNode), () => {
              const illumination = getSunAndSkyScalarIlluminance(
                correctedPosition,
                atmosphere.sunDirectionECEF
              ).toConst()
              const cosTheta = rayDirectionUnit
                .negate()
                .dot(atmosphere.sunDirectionECEF)
                .toConst()
              const surfaceNormal = correctedPosition.normalize().toConst()
              const mediaWeight = density.div(densitySum.add(EPSILON)).toConst()
              const skyGradient = heightFraction
                .mul(0.5)
                .add(0.5)
                .dot(mediaWeight)
                .toConst()
              const skyLight = illumination
                .get('skyIlluminance')
                .mul(float(RECIPROCAL_PI4))
                .mul(skyGradient)
                .mul(clouds.skyLightScaleNode)
                .toConst()

              const scattering = densitySum
                .mul(clouds.scatteringCoefficientNode)
                .toConst()
              const extinction = densitySum
                .mul(
                  clouds.scatteringCoefficientNode.add(
                    clouds.absorptionCoefficientNode
                  )
                )
                .max(clouds.minExtinctionNode)
                .toConst()
              const sunOpticalDepth = float(0.5).toVar()
              const beerShadowOpticalDepth = float(0).toVar()
              const secondaryDistanceWorld = float(0).toVar()
              const secondarySampleDistanceWorld = float(0).toVar()
              If(clouds.maxIterationCountToSunNode.greaterThan(0), () => {
                const secondaryIterationCount =
                  clouds.maxIterationCountToSunNode.toFloat().toConst()
                const secondaryStepSizeWorld = clouds.minSecondaryStepSizeNode
                  .div(secondaryIterationCount.max(EPSILON))
                  .toVar()
                secondaryDistanceWorld.assign(
                  secondaryStepSizeWorld.mul(stochasticJitter)
                )
                sunOpticalDepth.assign(0)

                Loop(
                  { start: 0, end: WEBGPU_MAX_SECONDARY_STEPS, condition: '<' },
                  ({ i }) => {
                    If(
                      float(i).greaterThanEqual(
                        clouds.maxIterationCountToSunNode.toFloat()
                      ),
                      () => {
                        Break()
                      }
                    )

                    secondarySampleDistanceWorld.assign(secondaryDistanceWorld)
                    const secondaryPositionUnit = correctedPosition
                      .add(
                        atmosphere.sunDirectionECEF.mul(
                          secondarySampleDistanceWorld.mul(
                            atmosphere.worldToUnit
                          )
                        )
                      )
                      .toConst()
                    const secondaryExtinction = sampleExtinctionAt(
                      secondaryPositionUnit,
                      mipLevel,
                      stochasticJitter,
                      float(1)
                    ).toConst()

                    sunOpticalDepth.addAssign(
                      secondaryExtinction.mul(secondaryStepSizeWorld)
                    )
                    secondaryDistanceWorld.addAssign(secondaryStepSizeWorld)
                    secondaryStepSizeWorld.mulAssign(
                      clouds.secondaryStepScaleNode
                    )
                  }
                )
              })
              if (cloudsShadowNode != null) {
                If(height.lessThan(clouds.shadowMaxHeightNode), () => {
                  const shadowFilterRadius = clouds.maxShadowFilterRadiusNode
                    .mul(
                      float(1).sub(
                        remapClamp(
                          atmosphere.sunDirectionECEF.dot(surfaceNormal),
                          float(0),
                          float(0.1)
                        )
                      )
                    )
                    .toConst()
                  const shadowSamplePositionWorld = atmosphere.matrixECEFToWorld
                    .mul(
                      vec4(
                        correctedPosition
                          .sub(altitudeCorrectionUnit)
                          .mul(unitToWorld),
                        1
                      )
                    )
                    .xyz
                    .toConst()
                  beerShadowOpticalDepth.assign(
                    cloudsShadowNode
                      .sampleOpticalDepth(
                        shadowSamplePositionWorld,
                        secondarySampleDistanceWorld,
                        null,
                        {
                          filterRadiusNode: shadowFilterRadius,
                          cascadeJitterNode: stochasticJitter
                        }
                      )
                      .toConst()
                  )
                  sunOpticalDepth.addAssign(beerShadowOpticalDepth)
                })
              }
              const sunLight = illumination
                .get('sunIlluminance')
                .mul(approximateMultipleScattering(sunOpticalDepth, cosTheta))
                .toConst()
              const powderFactor = float(1)
                .sub(
                  clouds.powderScaleNode.mul(
                    exp(extinction.negate().mul(clouds.powderExponentNode))
                  )
                )
                .max(0)
                .toConst()
              const groundBounce = vec3(0).toVar()
              If(
                clouds.groundBounceScaleNode
                  .greaterThan(EPSILON)
                  .and(clouds.maxIterationCountToGroundNode.greaterThan(0))
                  .and(mipLevel.lessThan(0.5)),
                () => {
                  const lowerCloudWeight = float(1)
                    .sub(
                      remapClamp(
                        height,
                        clouds.minHeightNode,
                        clouds.maxHeightNode
                      )
                    )
                    .pow2()
                    .toConst()
                  groundBounce.assign(
                    approximateGroundRadiance(
                      correctedPosition,
                      surfaceNormal,
                      height,
                      mipLevel,
                      stochasticJitter
                    )
                      .mul(RECIPROCAL_PI4)
                      .mul(clouds.groundBounceScaleNode)
                      .mul(lowerCloudWeight)
                  )
                }
              )
              const lighting = skyLight
                .add(sunLight)
                .add(groundBounce)
                .mul(powderFactor)
                .toConst()
              const sampleTransmittance = exp(
                extinction.negate().mul(rayStepSizeWorld)
              ).toConst()
              const contribution = transmittance
                .mul(sampleTransmittance.oneMinus())
                .mul(scattering.div(extinction.add(EPSILON)))
                .toConst()
              const nextTransmittance = transmittance
                .mul(sampleTransmittance)
                .toConst()

              radiance.addAssign(lighting.mul(contribution))
              transmittanceWeightedDepth.addAssign(
                rayDistance.mul(nextTransmittance)
              )
              transmittanceWeightSum.addAssign(nextTransmittance)
              transmittance.assign(nextTransmittance)
            })

            stepSizeWorld.mulAssign(clouds.perspectiveStepScaleNode)
            rayDistance.addAssign(stepSizeWorld.mul(atmosphere.worldToUnit))
          }
        )

        const alpha = remapClamp(
          transmittance,
          float(1),
          clouds.minTransmittanceNode
        ).toConst()
        const frontDistance = transmittanceWeightedDepth
          .div(transmittanceWeightSum.add(EPSILON))
          .toConst()
        sceneColor.assign(vec4(baseSceneColor.mul(transmittance), sceneColor.a))
        // Pack alpha, remaining transmittance, transmittance-weighted mean
        // depth and the accumulated weight for downstream nodes.
        transmittanceDepth
          .assign(
            vec4(
              alpha,
              transmittance,
              transmittanceWeightedDepth.div(
                transmittanceWeightSum.add(EPSILON)
              ),
              transmittanceWeightSum
            )
          )
          .toConst()

        If(transmittanceWeightSum.greaterThan(EPSILON), () => {
            const frontPositionUnit = cameraPositionUnit
              .add(rayDirectionUnit.mul(frontDistance))
              .toConst()
            const frontDistanceWorld = frontDistance.mul(unitToWorld).toConst()
            if (marchShadowLengthWorld != null) {
              const cloudShadowNearFarWorld = shadowNearFarWorld.toVar()
              cloudShadowNearFarWorld.y.assign(
                mix(
                  cloudShadowNearFarWorld.y,
                  min(frontDistanceWorld, cloudShadowNearFarWorld.y),
                  alpha
                )
              )
              If(
                cloudShadowNearFarWorld.x
                  .greaterThanEqual(0)
                  .and(
                    cloudShadowNearFarWorld.y.greaterThan(
                      cloudShadowNearFarWorld.x
                    )
                  ),
                () => {
                  shadowLengthWorld.assign(
                    marchShadowLengthWorld(
                      cameraWorld.add(
                        rayDirectionWorld.mul(cloudShadowNearFarWorld.x)
                      ),
                      rayDirectionWorld,
                      cloudShadowNearFarWorld.y.sub(cloudShadowNearFarWorld.x),
                      stochasticJitter
                    )
                  )
                }
              )
            }
            const luminanceTransfer = getSkyLuminanceToPoint(
              cameraPositionUnit,
              frontPositionUnit,
              shadowLengthWorld.mul(atmosphere.worldToUnit),
              atmosphere.sunDirectionECEF
            ).toConst()
            sceneColor.assign(
              vec4(
                sceneColor.rgb.add(
                  radiance
                    .mul(luminanceTransfer.get('transmittance'))
                    .add(luminanceTransfer.get('luminance').mul(alpha))
                ),
                sceneColor.a
              )
            )

            hazeNearFarWorld.y.assign(
              mix(
                hazeNearFarWorld.y,
                min(frontDistanceWorld, hazeNearFarWorld.y),
                alpha
              )
            )
            const frontPositionWorld = cameraWorld
              .add(rayDirectionWorld.mul(frontDistanceWorld))
              .toConst()
            const localWeatherMotionWorld = approximateLocalWeatherMotionWorld(
              frontPositionUnit
            ).toConst()
            const frontNormalWorld = frontPositionWorld.normalize().toConst()
            const currentEvolutionWorld = frontNormalWorld
              .negate()
              .mul(currentLocalWeatherSpeed)
              .mul(2e4)
              .toConst()
            const previousEvolutionWorld = frontNormalWorld
              .negate()
              .mul(previousLocalWeatherSpeed)
              .mul(2e4)
              .toConst()
            const previousFrontPositionWorld = frontPositionWorld
              .add(shapeMotionWorld)
              .add(shapeDetailMotionWorld)
              .add(localWeatherMotionWorld)
              .add(currentEvolutionWorld.sub(previousEvolutionWorld))
              .toConst()
            const frontPositionView = currentView
              .mul(vec4(frontPositionWorld, 1))
              .xyz.toConst()

            const currentClip = currentProjection
              .mul(currentView)
              .mul(vec4(frontPositionWorld, 1))
              .toConst()
            const previousClip = previousProjection
              .mul(previousView)
              .mul(vec4(previousFrontPositionWorld, 1))
              .toConst()
            const currentNdc = currentClip.xyz
              .div(currentClip.w.abs().max(EPSILON))
              .toConst()
            const previousNdc = previousClip.xyz
              .div(previousClip.w.abs().max(EPSILON))
              .toConst()
            cloudVelocity.assign(vec4(currentNdc.sub(previousNdc), 1))

            this.resolvedDepthNode.assign(encodeDepth(frontPositionView.z))
          }
        )
      })

      const hazeDistanceWorld = max(
        float(0),
        hazeNearFarWorld.y.sub(hazeNearFarWorld.x)
      ).toConst()

      If(hazeDistanceWorld.greaterThan(EPSILON), () => {
        const haze = approximateHaze(
          cameraPositionUnit.add(rayDirectionUnit.mul(near.mul(atmosphere.worldToUnit))),
          rayDirectionUnit,
          hazeDistanceWorld,
          sunCosTheta,
          shadowLengthWorld
        ).toConst()
        sceneColor.assign(
          vec4(
            mix(sceneColor.rgb, haze.rgb, haze.a),
            sceneColor.a.mul(haze.a.oneMinus()).add(haze.a)
          )
        )
      })

      effectColor.assign(
        vec4(
          sceneColor.rgb.sub(baseSceneColor.mul(transmittanceDepth.y)),
          sceneColor.a
        )
      )

      return mrt({
        output: sceneColor,
        effectColor,
        transmittanceDepth,
        velocity: cloudVelocity
      })
    })()

    // `Fn()` returns a shader-call node. Mark it explicitly so NodeMaterial
    // does not wrap the MRT result into a single vec4 output.
    ;(
      fragmentNode as Node & {
        isOutputStructNode?: boolean
      }
    ).isOutputStructNode = true

    return fragmentNode
  }
}

export const clouds = (
  ...args: ConstructorParameters<typeof CloudsNode>
): CloudsNode => new CloudsNode(...args)
