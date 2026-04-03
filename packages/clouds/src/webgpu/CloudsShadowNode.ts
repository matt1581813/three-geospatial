import {
  HalfFloatType,
  LinearFilter,
  Matrix3,
  Matrix4,
  NearestFilter,
  Quaternion,
  RenderTarget,
  RGBAFormat,
  Vector2,
  Vector3,
  Vector4,
  type Camera,
  type PerspectiveCamera,
  type Texture
} from 'three'
import { hash } from 'three/src/nodes/core/NodeUtils.js'
import {
  Break,
  cos,
  Continue,
  dot,
  exp,
  float,
  Fn,
  If,
  ivec2,
  Loop,
  max,
  min,
  mix,
  mrt,
  remapClamp,
  screenCoordinate,
  select,
  sin,
  sqrt,
  texture3D,
  texture,
  textureLevel,
  uniform,
  uv,
  vec2,
  vec3,
  vec4
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
  type AtmosphereContext
} from '@takram/three-atmosphere/webgpu'
import {
  FnLayout,
  FnVar,
  outputTexture,
  interleavedGradientNoise,
  raySphereIntersection,
  raySpheresIntersections,
  viewMatrix,
  type Node
} from '@takram/three-geospatial/webgpu'

import { CascadedShadowMaps } from '../CascadedShadowMaps'
import {
  CloudsContext,
  fallbackLocalWeatherTexture,
  fallbackShapeDetailTexture,
  fallbackShapeTexture,
  fallbackTurbulenceTexture,
  getCloudsContext
} from './CloudsContext'
import { CloudsRenderTargets } from './CloudsRenderTargets'
import { CloudsTemporalState } from './CloudsTemporalState'
import { WEBGPU_MAX_PRIMARY_STEPS } from './qualityPresets'

const { resetRendererState, restoreRendererState } = RendererUtils

const EPSILON = 1e-6
const MAX_CLOUD_SHADOW_CASCADES = 4
const SHADOW_MIP_LEVELS = [0, 0.5, 1, 2] as const
export const WEBGPU_CLOUD_BODY_MAX_FILTER_RADIUS = 6
export const CLOUD_SHADOW_TEMPORAL_ALPHA = 0.01
export const CLOUD_SHADOW_VARIANCE_GAMMA = 1
const CLOUD_SHADOW_CAMERA_CUT_POSITION_THRESHOLD = 1_000
const CLOUD_SHADOW_CAMERA_CUT_ROTATION_THRESHOLD = Math.PI / 12
const CLOUD_SHADOW_CAMERA_CUT_PROJECTION_THRESHOLD = 1e-3
const CLOUD_BODY_SHADOW_PCF_OFFSETS = [
  [-0.613392, 0.617481],
  [0.170019, -0.040254],
  [-0.299417, 0.791925],
  [0.64568, 0.49321],
  [-0.651784, 0.717887],
  [0.421003, 0.02707],
  [-0.817194, -0.271096],
  [-0.705374, -0.668203]
] as const
const HIGH_FREQUENCY_FADE_START = 20_000
const HIGH_FREQUENCY_FADE_END = 120_000
const varianceOffsets = [
  /*#__PURE__*/ ivec2(-1, -1),
  /*#__PURE__*/ ivec2(-1, 1),
  /*#__PURE__*/ ivec2(1, -1),
  /*#__PURE__*/ ivec2(1, 1),
  /*#__PURE__*/ ivec2(1, 0),
  /*#__PURE__*/ ivec2(0, -1),
  /*#__PURE__*/ ivec2(0, 1),
  /*#__PURE__*/ ivec2(-1, 0)
] as const
const closestNeighborOffsets = [
  /*#__PURE__*/ ivec2(-1, -1),
  /*#__PURE__*/ ivec2(-1, 0),
  /*#__PURE__*/ ivec2(-1, 1),
  /*#__PURE__*/ ivec2(0, -1),
  /*#__PURE__*/ ivec2(0, 0),
  /*#__PURE__*/ ivec2(0, 1),
  /*#__PURE__*/ ivec2(1, -1),
  /*#__PURE__*/ ivec2(1, 0),
  /*#__PURE__*/ ivec2(1, 1)
] as const
const viewportScratch = /*#__PURE__*/ new Vector4()
const scissorScratch = /*#__PURE__*/ new Vector4()
const rotationScratch = /*#__PURE__*/ new Matrix3()
const vectorScratch1 = /*#__PURE__*/ new Vector3()
const vectorScratch2 = /*#__PURE__*/ new Vector3()
const vectorScratch3 = /*#__PURE__*/ new Vector3()
const cameraPositionScratch = /*#__PURE__*/ new Vector3()
const cameraQuaternionScratch = /*#__PURE__*/ new Quaternion()

type CloudsShadowCurrentTextureName = 'shadow' | 'depthVelocity'

export interface CloudShadowAtlasViewport {
  x: number
  y: number
  width: number
  height: number
}

export function clampCloudShadowCascadeCount(value: number): number {
  return Math.min(Math.max(Math.round(value), 1), MAX_CLOUD_SHADOW_CASCADES)
}

export function getCloudShadowAtlasSize(
  mapSize: Vector2,
  cascadeCount: number
): Vector2 {
  const safeCascadeCount = clampCloudShadowCascadeCount(cascadeCount)
  return new Vector2(
    Math.max(Math.round(mapSize.x), 1) * safeCascadeCount,
    Math.max(Math.round(mapSize.y), 1)
  )
}

export function getCloudShadowAtlasViewport(
  cascadeIndex: number,
  mapSize: Vector2,
  cascadeCount: number
): CloudShadowAtlasViewport {
  const safeCascadeCount = clampCloudShadowCascadeCount(cascadeCount)
  const width = Math.max(Math.round(mapSize.x), 1)
  const height = Math.max(Math.round(mapSize.y), 1)
  const index = Math.min(Math.max(Math.round(cascadeIndex), 0), safeCascadeCount - 1)

  return {
    x: index * width,
    y: 0,
    width,
    height
  }
}

export function getCloudShadowCascadeDepth(
  viewDepth: number,
  near: number,
  far: number
): number {
  return (viewDepth - near) / Math.max(far - near, EPSILON)
}

export function getCloudShadowCascadeFadeWidth(cascadeBoundaryDepth: number): number {
  return Math.min(Math.max(cascadeBoundaryDepth * cascadeBoundaryDepth * 0.5, 0.004), 0.06)
}

export function resolveCloudShadowCascadeIndex(
  cascadeDepth: number,
  intervals: readonly Vector2[],
  cascadeCount: number
): number {
  const safeCascadeCount = clampCloudShadowCascadeCount(cascadeCount)
  for (let index = 0; index < safeCascadeCount - 1; ++index) {
    const interval = intervals[index]
    if (cascadeDepth >= interval.x && cascadeDepth < interval.y) {
      return index
    }
  }
  return safeCascadeCount - 1
}

export function getCloudShadowCascadeBlendWeights(
  cascadeDepth: number,
  intervals: readonly Vector2[],
  cascadeCount: number
): readonly [number, number, number, number] {
  const safeCascadeCount = clampCloudShadowCascadeCount(cascadeCount)
  const split01 = intervals[0]?.y ?? 1
  const split12 = intervals[1]?.y ?? 1
  const split23 = intervals[2]?.y ?? 1
  const fade01 = getCloudShadowCascadeFadeWidth(split01)
  const fade12 = getCloudShadowCascadeFadeWidth(split12)
  const fade23 = getCloudShadowCascadeFadeWidth(split23)
  const transition01 =
    safeCascadeCount > 1
      ? Math.min(
          Math.max(
            (cascadeDepth - (split01 - fade01 * 0.5)) / Math.max(fade01, EPSILON),
            0
          ),
          1
        )
      : 0
  const transition12 =
    safeCascadeCount > 2
      ? Math.min(
          Math.max(
            (cascadeDepth - (split12 - fade12 * 0.5)) / Math.max(fade12, EPSILON),
            0
          ),
          1
        )
      : 0
  const transition23 =
    safeCascadeCount > 3
      ? Math.min(
          Math.max(
            (cascadeDepth - (split23 - fade23 * 0.5)) / Math.max(fade23, EPSILON),
            0
          ),
          1
        )
      : 0

  const weight0 = 1 - transition01
  const weight1 =
    safeCascadeCount > 1
      ? safeCascadeCount > 2
        ? transition01 * (1 - transition12)
        : transition01
      : 0
  const weight2 =
    safeCascadeCount > 2
      ? safeCascadeCount > 3
        ? transition12 * (1 - transition23)
        : transition12
      : 0
  const weight3 = safeCascadeCount > 3 ? transition23 : 0

  return [weight0, weight1, weight2, weight3]
}

export function computeCloudShadowOpticalDepth(
  distanceToTop: number,
  distanceOffset: number,
  frontDepth: number,
  meanExtinction: number,
  maxOpticalDepth: number,
  maxOpticalDepthTail: number
): number {
  const distanceToFront = Math.max(
    0,
    distanceToTop - distanceOffset - frontDepth
  )
  return Math.min(
    Math.max(maxOpticalDepth, 0) + Math.max(maxOpticalDepthTail, 0),
    Math.max(meanExtinction, 0) * distanceToFront
  )
}

export function computeCloudShadowOpticalDepthTail(
  stepSizeWorld: number,
  sampleCount: number,
  opticalDepthTailScale: number
): number {
  return Math.min(
    opticalDepthTailScale * stepSizeWorld * Math.exp(1 - sampleCount),
    stepSizeWorld * 0.5
  )
}

export function computeCloudShadowFilterRadius(
  sunDotSurfaceNormal: number,
  maxFilterRadius = WEBGPU_CLOUD_BODY_MAX_FILTER_RADIUS
): number {
  return (
    Math.max(maxFilterRadius, 0) *
    (1 -
      Math.min(Math.max(sunDotSurfaceNormal / 0.1, 0), 1))
  )
}

const getGlobeUv = /*#__PURE__*/ Fn(([position]: [Node<'vec3'>]) => {
  const n = position.normalize().toConst()
  const f = n.abs().toConst()
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
  const uvCoord = vec2().toVar()

  uvCoord.x.assign(
    float(1.5)
      .add(m2.x)
      .sub(m2.y)
      .sub(m2.x.mul(-24).add(q.pow2()).sqrt().mul(0.5))
      .sqrt()
      .mul(select(m.x.greaterThan(0), 1, -1))
  )
  uvCoord.y.assign(float(6).div(float(3).sub(uvCoord.x.pow2())).sqrt().mul(m.y))

  return uvCoord.mul(0.5).add(0.5)
})

function isPerspectiveCamera(camera: Camera): camera is PerspectiveCamera {
  return camera.isPerspectiveCamera === true
}

function getMaxMatrixDelta(a: Matrix4, b: Matrix4): number {
  let delta = 0
  for (let i = 0; i < 16; ++i) {
    delta = Math.max(delta, Math.abs(a.elements[i] - b.elements[i]))
  }
  return delta
}

const clipAABB = /*#__PURE__*/ FnLayout({
  name: 'clipAABB',
  type: 'vec4',
  inputs: [
    { name: 'current', type: 'vec4' },
    { name: 'history', type: 'vec4' },
    { name: 'minColor', type: 'vec4' },
    { name: 'maxColor', type: 'vec4' }
  ]
})(([current, history, minColor, maxColor]) => {
  const pClip = maxColor.rgb.add(minColor.rgb).mul(0.5).toConst()
  const eClip = maxColor.rgb.sub(minColor.rgb).mul(0.5).add(1e-7).toConst()
  const vClip = history.sub(vec4(pClip, current.a)).toConst()
  const vUnit = vClip.xyz.div(eClip).toConst()
  const absUnit = vUnit.abs().toConst()
  const maxUnit = max(absUnit.x, max(absUnit.y, absUnit.z)).toConst()

  return select(
    maxUnit.greaterThan(1),
    vec4(pClip, current.a).add(vClip.div(maxUnit)),
    history
  )
})

const varianceClippingTile = /*#__PURE__*/ FnVar(
  (
    inputNode: TextureNode,
    coord: Node<'ivec2'>,
    tileOrigin: Node<'ivec2'>,
    tileMaxCoord: Node<'ivec2'>,
    current: Node<'vec4'>,
    history: Node<'vec4'>,
    gamma: Node<'float'>
  ): Node<'vec4'> => {
    const moment1 = current.toVar()
    const moment2 = current.pow2().toVar()

    for (const offset of varianceOffsets) {
      const neighborCoord = coord.add(offset).clamp(tileOrigin, tileMaxCoord).toConst()
      const neighbor = inputNode.load(neighborCoord).toConst()
      moment1.addAssign(neighbor)
      moment2.addAssign(neighbor.pow2())
    }

    const sampleCount = varianceOffsets.length + 1
    const mean = moment1.div(sampleCount).toConst()
    const variance = sqrt(moment2.div(sampleCount).sub(mean.pow2()).max(0))
      .mul(gamma)
      .toConst()
    const minColor = mean.sub(variance).toConst()
    const maxColor = mean.add(variance).toConst()

    return clipAABB(mean.clamp(minColor, maxColor), history, minColor, maxColor)
  }
)

const sizeScratch = /*#__PURE__*/ new Vector2()

export class CloudsShadowNode extends TempNode {
  static override get type(): string {
    return 'CloudsShadowNode'
  }

  camera: PerspectiveCamera

  private readonly shadowMaps = new CascadedShadowMaps({
    cascadeCount: 3,
    mapSize: new Vector2(512, 512),
    splitLambda: 0.6
  })
  private readonly currentRenderTargets =
    new CloudsRenderTargets<CloudsShadowCurrentTextureName>(null, {
      colorAttachments: ['shadow', 'depthVelocity']
    })
  private resolveRenderTarget = this.createRenderTarget('Resolve')
  private historyRenderTarget = this.createRenderTarget('History')
  private readonly textureNode = outputTexture(this, this.resolveRenderTarget.texture)
  private readonly currentMaterial = new NodeMaterial()
  private readonly resolveMaterial = new NodeMaterial()
  private readonly currentMesh = new QuadMesh(this.currentMaterial)
  private readonly resolveMesh = new QuadMesh(this.resolveMaterial)
  private readonly historyNode = texture(this.historyRenderTarget.texture)

  private readonly localWeatherNode = texture(fallbackLocalWeatherTexture)
  private readonly shapeTextureNode = texture3D(fallbackShapeTexture)
  private readonly shapeDetailTextureNode = texture3D(fallbackShapeDetailTexture)
  private readonly turbulenceTextureNode = texture(fallbackTurbulenceTexture)

  private readonly currentInverseShadowMatrixNode = uniform(new Matrix4()).setName(
    'cloudsShadowInverseMatrix'
  )
  private readonly currentReprojectionMatrixNode = uniform(new Matrix4()).setName(
    'cloudsShadowReprojectionMatrix'
  )
  private readonly currentShadowMipLevelNode = uniform(0).setName(
    'cloudsShadowMipLevel'
  )
  private readonly shadowCascadeCountNode = uniform(3, 'int').setName(
    'cloudsShadowCascadeCount'
  )
  private readonly shadowNormalBiasNode = uniform(40).setName(
    'cloudsShadowNormalBias'
  )
  private readonly shadowNearNode = uniform(0.1).setName('cloudsShadowNear')
  private readonly shadowFarNode = uniform(1000).setName('cloudsShadowFar')
  private readonly shadowTileTexelSizeNode = uniform(new Vector2(1, 1)).setName(
    'cloudsShadowTileTexelSize'
  )
  private readonly shadowTileSizeNode = uniform(new Vector2(1, 1)).setName(
    'cloudsShadowTileSize'
  )
  private readonly shadowIntervalNodes = Array.from(
    { length: MAX_CLOUD_SHADOW_CASCADES },
    (_, index) =>
      uniform(new Vector2(index, Number.POSITIVE_INFINITY)).setName(
        `cloudsShadowInterval${index}`
      )
  )
  private readonly shadowMatrixNodes = Array.from(
    { length: MAX_CLOUD_SHADOW_CASCADES },
    (_, index) =>
      uniform(new Matrix4()).setName(`cloudsShadowMatrix${index}`)
  )

  private readonly maxIterationCountNode = uniform(50, 'int').setName(
    'cloudsShadowMaxIterationCount'
  )
  private readonly minStepSizeNode = uniform(100).setName(
    'cloudsShadowMinStepSize'
  )
  private readonly maxStepSizeNode = uniform(1000).setName(
    'cloudsShadowMaxStepSize'
  )
  private readonly minDensityNode = uniform(1e-5).setName(
    'cloudsShadowMinDensity'
  )
  private readonly minExtinctionNode = uniform(1e-5).setName(
    'cloudsShadowMinExtinction'
  )
  private readonly minTransmittanceNode = uniform(1e-4).setName(
    'cloudsShadowMinTransmittance'
  )
  private readonly opticalDepthTailScaleNode = uniform(2).setName(
    'cloudsShadowOpticalDepthTailScale'
  )
  private readonly resolveHistoryWeightNode = uniform(0).setName(
    'cloudsShadowResolveHistoryWeight'
  )

  private rendererState?: RendererUtils.RendererState
  private cloudsContext?: CloudsContext
  private atmosphereContext?: AtmosphereContext
  private readonly temporalState = new CloudsTemporalState()
  private readonly previousShadowMatrices = Array.from(
    { length: MAX_CLOUD_SHADOW_CASCADES },
    () => new Matrix4()
  )
  private readonly previousCameraPosition = new Vector3()
  private readonly previousCameraQuaternion = new Quaternion()
  private readonly previousProjectionMatrix = new Matrix4()
  private previousFrameValid = false
  private cameraPoseValid = false

  constructor(camera: Camera) {
    super('vec4')

    if (!isPerspectiveCamera(camera)) {
      throw new Error('CloudsShadowNode currently requires a PerspectiveCamera.')
    }
    this.camera = camera

    this.currentRenderTargets.getTexture('depthVelocity').minFilter = NearestFilter
    this.currentRenderTargets.getTexture('depthVelocity').magFilter = NearestFilter
    this.currentMaterial.name = 'CloudsShadowNode.Current'
    this.resolveMaterial.name = 'CloudsShadowNode.Resolve'

    this.updateBeforeType = NodeUpdateType.FRAME
  }

  override customCacheKey(): number {
    return hash(this.camera.id)
  }

  private createRenderTarget(name: string): RenderTarget {
    const renderTarget = new RenderTarget(1, 1, {
      depthBuffer: false,
      type: HalfFloatType,
      format: RGBAFormat
    })
    renderTarget.texture.name = `CloudsShadowNode.${name}`
    renderTarget.texture.minFilter = LinearFilter
    renderTarget.texture.magFilter = LinearFilter
    renderTarget.texture.generateMipmaps = false
    return renderTarget
  }

  getTexture(): Texture {
    return this.textureNode.value
  }

  getTextureNode(): TextureNode {
    return this.textureNode
  }

  setContexts(cloudsContext: CloudsContext, atmosphereContext: AtmosphereContext): void {
    this.cloudsContext = cloudsContext
    this.atmosphereContext = atmosphereContext
  }

  private sampleData(
    positionWorldNode: Node<'vec3'>,
    normalWorldNode: Node<'vec3'> | null = null,
    { filtered = true }: { filtered?: boolean } = {}
  ): Node<'vec4'> {
    const atlasNode = this.getTextureNode()
    const camera = this.camera

    const projectCascade = (
      matrixNode: Node<'mat4'>,
      cascadeIndex: number,
      samplePositionWorld: Node<'vec3'>,
      filterScaleNode: Node<'float'>
    ): Node<'vec4'> => {
      const clip = matrixNode.mul(vec4(samplePositionWorld, 1)).toConst()
      const ndc = clip.xyz.div(clip.w.abs().max(EPSILON)).toConst()
      const localUv = ndc.xy.mul(0.5).add(0.5).toConst()
      const inBounds = localUv
        .greaterThanEqual(vec2(0))
        .all()
        .and(localUv.lessThanEqual(vec2(1)).all())
        .toConst()
      const texelSize = this.shadowTileTexelSizeNode.mul(filterScaleNode).toConst()
      const minLocalUv = texelSize.mul(0.5).toConst()
      const maxLocalUv = vec2(1).sub(minLocalUv).toConst()

      const centerUv = vec2(
        localUv.x
          .clamp(minLocalUv.x, maxLocalUv.x)
          .add(float(cascadeIndex))
          .div(this.shadowCascadeCountNode.toFloat()),
        localUv.y.clamp(minLocalUv.y, maxLocalUv.y)
      ).toConst()
      const leftUv = vec2(
        localUv.x
          .sub(texelSize.x)
          .clamp(minLocalUv.x, maxLocalUv.x)
          .add(float(cascadeIndex))
          .div(this.shadowCascadeCountNode.toFloat()),
        localUv.y.clamp(minLocalUv.y, maxLocalUv.y)
      ).toConst()
      const rightUv = vec2(
        localUv.x
          .add(texelSize.x)
          .clamp(minLocalUv.x, maxLocalUv.x)
          .add(float(cascadeIndex))
          .div(this.shadowCascadeCountNode.toFloat()),
        localUv.y.clamp(minLocalUv.y, maxLocalUv.y)
      ).toConst()
      const topUv = vec2(
        localUv.x
          .clamp(minLocalUv.x, maxLocalUv.x)
          .add(float(cascadeIndex))
          .div(this.shadowCascadeCountNode.toFloat()),
        localUv.y.add(texelSize.y).clamp(minLocalUv.y, maxLocalUv.y)
      ).toConst()
      const bottomUv = vec2(
        localUv.x
          .clamp(minLocalUv.x, maxLocalUv.x)
          .add(float(cascadeIndex))
          .div(this.shadowCascadeCountNode.toFloat()),
        localUv.y.sub(texelSize.y).clamp(minLocalUv.y, maxLocalUv.y)
      ).toConst()
      const topLeftUv = vec2(
        localUv.x
          .sub(texelSize.x)
          .clamp(minLocalUv.x, maxLocalUv.x)
          .add(float(cascadeIndex))
          .div(this.shadowCascadeCountNode.toFloat()),
        localUv.y.add(texelSize.y).clamp(minLocalUv.y, maxLocalUv.y)
      ).toConst()
      const topRightUv = vec2(
        localUv.x
          .add(texelSize.x)
          .clamp(minLocalUv.x, maxLocalUv.x)
          .add(float(cascadeIndex))
          .div(this.shadowCascadeCountNode.toFloat()),
        localUv.y.add(texelSize.y).clamp(minLocalUv.y, maxLocalUv.y)
      ).toConst()
      const bottomLeftUv = vec2(
        localUv.x
          .sub(texelSize.x)
          .clamp(minLocalUv.x, maxLocalUv.x)
          .add(float(cascadeIndex))
          .div(this.shadowCascadeCountNode.toFloat()),
        localUv.y.sub(texelSize.y).clamp(minLocalUv.y, maxLocalUv.y)
      ).toConst()
      const bottomRightUv = vec2(
        localUv.x
          .add(texelSize.x)
          .clamp(minLocalUv.x, maxLocalUv.x)
          .add(float(cascadeIndex))
          .div(this.shadowCascadeCountNode.toFloat()),
        localUv.y.sub(texelSize.y).clamp(minLocalUv.y, maxLocalUv.y)
      ).toConst()

      const filteredSample = filtered
        ? atlasNode
            .sample(centerUv)
            .mul(0.25)
            .add(atlasNode.sample(leftUv).mul(0.125))
            .add(atlasNode.sample(rightUv).mul(0.125))
            .add(atlasNode.sample(topUv).mul(0.125))
            .add(atlasNode.sample(bottomUv).mul(0.125))
            .add(atlasNode.sample(topLeftUv).mul(0.0625))
            .add(atlasNode.sample(topRightUv).mul(0.0625))
            .add(atlasNode.sample(bottomLeftUv).mul(0.0625))
            .add(atlasNode.sample(bottomRightUv).mul(0.0625))
            .toConst()
        : atlasNode.sample(centerUv).toConst()

      return select(inBounds, filteredSample, vec4(-1))
    }

    return Fn(() => {
      const normalOffset = normalWorldNode != null
        ? normalWorldNode.normalize().mul(this.shadowNormalBiasNode)
        : vec3(0)
      const samplePositionWorld = positionWorldNode.add(normalOffset).toConst()
      const depthView = viewMatrix(camera)
        .mul(vec4(samplePositionWorld, 1))
        .z.negate()
        .toConst()
      const cascadeDepth = depthView
        .sub(this.shadowNearNode)
        .div(max(this.shadowFarNode.sub(this.shadowNearNode), float(EPSILON)))
        .toConst()
      const receiverFilterScale = mix(
        float(2),
        float(9),
        cascadeDepth.clamp(0, 1).pow2()
      ).toConst()
      const sample0 = projectCascade(
        this.shadowMatrixNodes[0],
        0,
        samplePositionWorld,
        receiverFilterScale
      ).toConst()
      const sample1 = projectCascade(
        this.shadowMatrixNodes[1],
        1,
        samplePositionWorld,
        receiverFilterScale
      ).toConst()
      const sample2 = projectCascade(
        this.shadowMatrixNodes[2],
        2,
        samplePositionWorld,
        receiverFilterScale
      ).toConst()
      const sample3 = projectCascade(
        this.shadowMatrixNodes[3],
        3,
        samplePositionWorld,
        receiverFilterScale
      ).toConst()

      const valid0 = sample0.greaterThanEqual(0).toConst()
      const valid1 = sample1.greaterThanEqual(0).toConst()
      const valid2 = sample2.greaterThanEqual(0).toConst()
      const valid3 = sample3.greaterThanEqual(0).toConst()

      const hasCascade1 = this.shadowCascadeCountNode.greaterThan(1).toConst()
      const hasCascade2 = this.shadowCascadeCountNode.greaterThan(2).toConst()
      const hasCascade3 = this.shadowCascadeCountNode.greaterThan(3).toConst()

      const split01 = this.shadowIntervalNodes[0].y.toConst()
      const split12 = this.shadowIntervalNodes[1].y.toConst()
      const split23 = this.shadowIntervalNodes[2].y.toConst()
      const fade01 = split01
        .pow2()
        .mul(0.5)
        .clamp(0.004, 0.06)
        .toConst()
      const fade12 = split12
        .pow2()
        .mul(0.5)
        .clamp(0.004, 0.06)
        .toConst()
      const fade23 = split23
        .pow2()
        .mul(0.5)
        .clamp(0.004, 0.06)
        .toConst()

      const transition01 = select(
        hasCascade1,
        remapClamp(
          cascadeDepth,
          split01.sub(fade01.mul(0.5)),
          split01.add(fade01.mul(0.5))
        ),
        float(0)
      ).toConst()
      const transition12 = select(
        hasCascade2,
        remapClamp(
          cascadeDepth,
          split12.sub(fade12.mul(0.5)),
          split12.add(fade12.mul(0.5))
        ),
        float(0)
      ).toConst()
      const transition23 = select(
        hasCascade3,
        remapClamp(
          cascadeDepth,
          split23.sub(fade23.mul(0.5)),
          split23.add(fade23.mul(0.5))
        ),
        float(0)
      ).toConst()

      const weight0 = float(1).sub(transition01).toConst()
      const weight1 = select(
        hasCascade1,
        select(
          hasCascade2,
          transition01.mul(float(1).sub(transition12)),
          transition01
        ),
        float(0)
      ).toConst()
      const weight2 = select(
        hasCascade2,
        select(
          hasCascade3,
          transition12.mul(float(1).sub(transition23)),
          transition12
        ),
        float(0)
      ).toConst()
      const weight3 = select(hasCascade3, transition23, float(0)).toConst()

      const weightedSum = vec4(0)
        .add(select(valid0, sample0, vec4(0)).mul(weight0))
        .add(select(valid1, sample1, vec4(0)).mul(weight1))
        .add(select(valid2, sample2, vec4(0)).mul(weight2))
        .add(select(valid3, sample3, vec4(0)).mul(weight3))
        .toConst()
      const totalWeight = select(valid0, weight0, float(0))
        .add(select(valid1, weight1, float(0)))
        .add(select(valid2, weight2, float(0)))
        .add(select(valid3, weight3, float(0)))
        .toConst()
      const fallbackSample = select(
        valid0,
        sample0,
        select(
          valid1,
          sample1,
          select(valid2, sample2, select(valid3, sample3, vec4(0)))
        )
      ).toConst()

      return select(
        totalWeight.greaterThan(EPSILON),
        weightedSum.div(totalWeight),
        fallbackSample
      )
    })()
  }

  private sampleOpticalDepthData(
    positionWorldNode: Node<'vec3'>,
    filterRadiusNode: Node<'float'>,
    cascadeJitterNode: Node<'float'> | null = null
  ): Node<'vec4'> {
    const atlasNode = this.getTextureNode()
    const camera = this.camera

    const projectCascade = (
      matrixNode: Node<'mat4'>,
      cascadeIndex: number,
      samplePositionWorld: Node<'vec3'>
    ): Node<'vec4'> => {
      const clip = matrixNode.mul(vec4(samplePositionWorld, 1)).toConst()
      const ndc = clip.xyz.div(clip.w.abs().max(EPSILON)).toConst()
      const localUv = ndc.xy.mul(0.5).add(0.5).toConst()
      const inBounds = localUv
        .greaterThanEqual(vec2(0))
        .all()
        .and(localUv.lessThanEqual(vec2(1)).all())
        .toConst()
      const texelSize = this.shadowTileTexelSizeNode.toConst()
      const minLocalUv = texelSize.mul(0.5).toConst()
      const maxLocalUv = vec2(1).sub(minLocalUv).toConst()
      const radius = max(filterRadiusNode, float(0)).toConst()
      const radiusTexel = texelSize.mul(radius).toConst()
      const toAtlasUv = (sampleLocalUv: Node<'vec2'>): Node<'vec2'> =>
        vec2(
          sampleLocalUv.x
            .clamp(minLocalUv.x, maxLocalUv.x)
            .add(float(cascadeIndex))
            .div(this.shadowCascadeCountNode.toFloat()),
          sampleLocalUv.y.clamp(minLocalUv.y, maxLocalUv.y)
        ).toConst()

      const centerSample = atlasNode.sample(toAtlasUv(localUv)).toConst()
      const pcfSum = vec4(0).toVar()
      for (const [offsetX, offsetY] of CLOUD_BODY_SHADOW_PCF_OFFSETS) {
        const sampleUv = localUv
          .add(vec2(float(offsetX), float(offsetY)).mul(radiusTexel))
          .toConst()
        pcfSum.addAssign(atlasNode.sample(toAtlasUv(sampleUv)))
      }
      const pcfSample = pcfSum
        .div(float(CLOUD_BODY_SHADOW_PCF_OFFSETS.length))
        .toConst()
      const sampled = select(radius.greaterThan(0.1), pcfSample, centerSample).toConst()

      return select(inBounds, sampled, vec4(-1))
    }

    return Fn(() => {
      const samplePositionWorld = positionWorldNode.toConst()
      const depthView = viewMatrix(camera)
        .mul(vec4(samplePositionWorld, 1))
        .z.negate()
        .toConst()
      const cascadeDepth = depthView
        .sub(this.shadowNearNode)
        .div(max(this.shadowFarNode.sub(this.shadowNearNode), float(EPSILON)))
        .toConst()
      const sample0 = projectCascade(
        this.shadowMatrixNodes[0],
        0,
        samplePositionWorld
      ).toConst()
      const sample1 = projectCascade(
        this.shadowMatrixNodes[1],
        1,
        samplePositionWorld
      ).toConst()
      const sample2 = projectCascade(
        this.shadowMatrixNodes[2],
        2,
        samplePositionWorld
      ).toConst()
      const sample3 = projectCascade(
        this.shadowMatrixNodes[3],
        3,
        samplePositionWorld
      ).toConst()

      const valid0 = sample0.x.greaterThanEqual(0).toConst()
      const valid1 = sample1.x.greaterThanEqual(0).toConst()
      const valid2 = sample2.x.greaterThanEqual(0).toConst()
      const valid3 = sample3.x.greaterThanEqual(0).toConst()

      const hasCascade1 = this.shadowCascadeCountNode.greaterThan(1).toConst()
      const hasCascade2 = this.shadowCascadeCountNode.greaterThan(2).toConst()
      const hasCascade3 = this.shadowCascadeCountNode.greaterThan(3).toConst()
      const split01 = this.shadowIntervalNodes[0].y.toConst()
      const split12 = this.shadowIntervalNodes[1].y.toConst()
      const split23 = this.shadowIntervalNodes[2].y.toConst()
      const fade01 = split01
        .pow2()
        .mul(0.5)
        .clamp(0.004, 0.06)
        .toConst()
      const fade12 = split12
        .pow2()
        .mul(0.5)
        .clamp(0.004, 0.06)
        .toConst()
      const fade23 = split23
        .pow2()
        .mul(0.5)
        .clamp(0.004, 0.06)
        .toConst()

      const transition01 = select(
        hasCascade1,
        remapClamp(
          cascadeDepth,
          split01.sub(fade01.mul(0.5)),
          split01.add(fade01.mul(0.5))
        ),
        float(0)
      ).toConst()
      const transition12 = select(
        hasCascade2,
        remapClamp(
          cascadeDepth,
          split12.sub(fade12.mul(0.5)),
          split12.add(fade12.mul(0.5))
        ),
        float(0)
      ).toConst()
      const transition23 = select(
        hasCascade3,
        remapClamp(
          cascadeDepth,
          split23.sub(fade23.mul(0.5)),
          split23.add(fade23.mul(0.5))
        ),
        float(0)
      ).toConst()

      const weight0 = float(1).sub(transition01).toConst()
      const weight1 = select(
        hasCascade1,
        select(
          hasCascade2,
          transition01.mul(float(1).sub(transition12)),
          transition01
        ),
        float(0)
      ).toConst()
      const weight2 = select(
        hasCascade2,
        select(
          hasCascade3,
          transition12.mul(float(1).sub(transition23)),
          transition12
        ),
        float(0)
      ).toConst()
      const weight3 = select(hasCascade3, transition23, float(0)).toConst()

      const weightedSum = vec4(0)
        .add(select(valid0, sample0, vec4(0)).mul(weight0))
        .add(select(valid1, sample1, vec4(0)).mul(weight1))
        .add(select(valid2, sample2, vec4(0)).mul(weight2))
        .add(select(valid3, sample3, vec4(0)).mul(weight3))
        .toConst()
      const totalWeight = select(valid0, weight0, float(0))
        .add(select(valid1, weight1, float(0)))
        .add(select(valid2, weight2, float(0)))
        .add(select(valid3, weight3, float(0)))
        .toConst()
      const fallbackSample = select(
        valid0,
        sample0,
        select(
          valid1,
          sample1,
          select(valid2, sample2, select(valid3, sample3, vec4(0)))
        )
      ).toConst()

      return select(
        totalWeight.greaterThan(EPSILON),
        weightedSum.div(totalWeight),
        fallbackSample
      )
    })()
  }

  sampleOpticalDepth(
    positionWorldNode: Node<'vec3'>,
    distanceOffsetWorldNode: Node<'float'> = float(0),
    normalWorldNode: Node<'vec3'> | null = null,
    {
      filtered = false,
      filterRadiusNode = null,
      cascadeJitterNode = null
    }: {
      filtered?: boolean
      filterRadiusNode?: Node<'float'> | null
      cascadeJitterNode?: Node<'float'> | null
    } = {}
  ): Node<'float'> {
    const atmosphere = this.atmosphereContext
    const clouds = this.cloudsContext
    if (atmosphere == null || clouds == null) {
      throw new Error(
        'CloudsShadowNode.sampleOpticalDepth() requires AtmosphereContext and CloudsContext.'
      )
    }

    return Fn(() => {
      const samplePositionWorld = positionWorldNode
        .add(
          normalWorldNode != null
            ? normalWorldNode.normalize().mul(this.shadowNormalBiasNode)
            : vec3(0)
        )
        .toConst()
      const shadow = (
        filterRadiusNode != null
          ? this.sampleOpticalDepthData(
              samplePositionWorld,
              filterRadiusNode,
              cascadeJitterNode
            )
          : this.sampleData(samplePositionWorld, null, {
              filtered
            })
      ).toConst()
      const samplePositionUnit = atmosphere.matrixWorldToECEF
        .mul(vec4(samplePositionWorld, 1))
        .xyz.mul(atmosphere.worldToUnit)
        .add(atmosphere.altitudeCorrectionUnit)
        .toConst()
      const shadowTopRadius = atmosphere.bottomRadius
        .add(clouds.shadowMaxHeightNode.mul(atmosphere.worldToUnit))
        .toConst()
      const distanceToTopUnit = raySphereIntersection(
        samplePositionUnit,
        atmosphere.sunDirectionECEF.normalize(),
        vec3(0),
        shadowTopRadius
      )
        .y.max(0)
        .toConst()
      const distanceToTopWorld = distanceToTopUnit
        .div(atmosphere.worldToUnit)
        .toConst()
      const distanceToFront = distanceToTopWorld
        .sub(distanceOffsetWorldNode)
        .sub(shadow.x)
        .max(0)
        .toConst()

      return min(shadow.z.add(shadow.w), shadow.y.mul(distanceToFront))
        .max(0)
        .toConst()
    })()
  }

  sample(
    positionWorldNode: Node<'vec3'>,
    normalWorldNode: Node<'vec3'> | null = null
  ): Node<'float'> {
    return exp(
      this.sampleOpticalDepth(positionWorldNode, float(0), normalWorldNode, {
        filtered: true
      }).negate()
    )
      .saturate()
      .toConst()
  }

  private setSize(width: number, height: number): boolean {
    if (
      width === this.currentRenderTargets.renderTarget.width &&
      height === this.currentRenderTargets.renderTarget.height
    ) {
      return false
    }
    this.currentRenderTargets.setSize(width, height)
    this.resolveRenderTarget.setSize(width, height)
    this.historyRenderTarget.setSize(width, height)
    return true
  }

  private swapBuffers(): void {
    const resolveRenderTarget = this.resolveRenderTarget
    this.resolveRenderTarget = this.historyRenderTarget
    this.historyRenderTarget = resolveRenderTarget
    this.historyNode.value = resolveRenderTarget.texture
    this.textureNode.value = resolveRenderTarget.texture
  }

  private detectCameraCut(projectionMatrix: Matrix4): boolean {
    if (!this.cameraPoseValid) {
      return false
    }

    cameraPositionScratch.setFromMatrixPosition(this.camera.matrixWorld)
    cameraQuaternionScratch.setFromRotationMatrix(this.camera.matrixWorld)

    const positionDelta = cameraPositionScratch.distanceTo(this.previousCameraPosition)
    const rotationDelta =
      2 *
      Math.acos(
        Math.min(1, Math.abs(cameraQuaternionScratch.dot(this.previousCameraQuaternion)))
      )
    const projectionDelta = getMaxMatrixDelta(
      projectionMatrix,
      this.previousProjectionMatrix
    )

    return (
      positionDelta > CLOUD_SHADOW_CAMERA_CUT_POSITION_THRESHOLD ||
      rotationDelta > CLOUD_SHADOW_CAMERA_CUT_ROTATION_THRESHOLD ||
      projectionDelta > CLOUD_SHADOW_CAMERA_CUT_PROJECTION_THRESHOLD
    )
  }

  private cacheCameraPose(projectionMatrix: Matrix4): void {
    this.previousCameraPosition.setFromMatrixPosition(this.camera.matrixWorld)
    this.previousCameraQuaternion.setFromRotationMatrix(this.camera.matrixWorld)
    this.previousProjectionMatrix.copy(projectionMatrix)
    this.cameraPoseValid = true
  }

  override updateBefore({ renderer, deltaTime, frameId }: NodeFrame): void {
    if (renderer == null || this.cloudsContext == null) {
      return
    }

    const clouds = this.cloudsContext
    const atmosphere = this.atmosphereContext
    if (atmosphere == null) {
      return
    }

    clouds.advance(frameId, deltaTime)
    this.localWeatherNode.value = clouds.resolvedLocalWeatherTexture
    this.shapeTextureNode.value = clouds.resolvedShapeTexture
    this.shapeDetailTextureNode.value = clouds.resolvedShapeDetailTexture
    this.turbulenceTextureNode.value = clouds.resolvedTurbulenceTexture
    this.temporalState.observe(clouds)

    const shadow = clouds.shadow
    const cascadeCount = clampCloudShadowCascadeCount(shadow.cascadeCount)
    const mapSize = new Vector2(
      Math.max(Math.round(shadow.mapSize.x), 1),
      Math.max(Math.round(shadow.mapSize.y), 1)
    )
    this.shadowMaps.cascadeCount = cascadeCount
    this.shadowMaps.mapSize.copy(mapSize)
    this.shadowMaps.maxFar = shadow.maxFar
    this.shadowMaps.farScale = shadow.farScale
    this.shadowMaps.splitMode = shadow.splitMode
    this.shadowMaps.splitLambda = shadow.splitLambda
    this.shadowMaps.margin = shadow.margin
    this.shadowMaps.fade = shadow.fade

    this.shadowCascadeCountNode.value = cascadeCount
    this.shadowNearNode.value = this.camera.near
    this.shadowTileTexelSizeNode.value.set(1 / mapSize.x, 1 / mapSize.y)
    this.shadowTileSizeNode.value.copy(mapSize)
    this.maxIterationCountNode.value = Math.max(
      1,
      Math.min(Math.round(shadow.maxIterationCount), WEBGPU_MAX_PRIMARY_STEPS)
    )
    this.minStepSizeNode.value = shadow.minStepSize
    this.maxStepSizeNode.value = shadow.maxStepSize
    this.minDensityNode.value = shadow.minDensity
    this.minExtinctionNode.value = shadow.minExtinction
    this.minTransmittanceNode.value = shadow.minTransmittance
    this.opticalDepthTailScaleNode.value = shadow.opticalDepthTailScale

    const worldToECEFMatrix = atmosphere.matrixWorldToECEF.value
    const cameraPositionECEF = this.camera
      .getWorldPosition(vectorScratch1)
      .applyMatrix4(worldToECEFMatrix)
    const surfaceNormal = atmosphere.ellipsoid.getSurfaceNormal(
      cameraPositionECEF,
      vectorScratch2
    )
    const zenithAngle = atmosphere.sunDirectionECEF.value.dot(surfaceNormal)
    const distance = 1e6 + (1e3 - 1e6) * zenithAngle
    const ecefToWorldRotation = rotationScratch
      .setFromMatrix4(worldToECEFMatrix)
      .transpose()
    const sunDirectionWorld = vectorScratch3
      .copy(atmosphere.sunDirectionECEF.value)
      .applyMatrix3(ecefToWorldRotation)
      .normalize()

    const historyResetRequested =
      this.temporalState.consumeHistoryReset() ||
      this.detectCameraCut(this.camera.projectionMatrix)

    this.shadowMaps.update(this.camera, sunDirectionWorld, distance)
    this.shadowFarNode.value = this.shadowMaps.far

    for (let index = 0; index < MAX_CLOUD_SHADOW_CASCADES; ++index) {
      const cascade = this.shadowMaps.cascades[index]
      if (cascade != null && index < cascadeCount) {
        this.shadowIntervalNodes[index].value.copy(cascade.interval)
        this.shadowMatrixNodes[index].value.copy(cascade.matrix)
      } else {
        this.shadowIntervalNodes[index].value.set(0, 1e12)
        this.shadowMatrixNodes[index].value.identity()
      }
    }

    const atlasSize = getCloudShadowAtlasSize(mapSize, cascadeCount)
    const resized = this.setSize(atlasSize.x, atlasSize.y)
    const useHistory = this.previousFrameValid && !historyResetRequested && !resized
    this.resolveHistoryWeightNode.value = useHistory ? 1 : 0

    renderer.getViewport(viewportScratch)
    renderer.getScissor(scissorScratch)
    const scissorTest = renderer.getScissorTest()

    this.rendererState = resetRendererState(renderer, this.rendererState)
    renderer.setRenderTarget(this.currentRenderTargets.renderTarget)
    renderer.setViewport(0, 0, atlasSize.x, atlasSize.y)
    renderer.setScissor(0, 0, atlasSize.x, atlasSize.y)
    renderer.setScissorTest(false)
    renderer.setClearColor(0xffffff, 1)
    renderer.clear()

    for (let index = 0; index < cascadeCount; ++index) {
      const cascade = this.shadowMaps.cascades[index]
      const viewport = getCloudShadowAtlasViewport(index, mapSize, cascadeCount)
      this.currentInverseShadowMatrixNode.value.copy(cascade.inverseMatrix)
      this.currentReprojectionMatrixNode.value.copy(
        useHistory ? this.previousShadowMatrices[index] : cascade.matrix
      )
      this.currentShadowMipLevelNode.value = SHADOW_MIP_LEVELS[index] ?? SHADOW_MIP_LEVELS.at(-1)!

      renderer.setViewport(viewport.x, viewport.y, viewport.width, viewport.height)
      renderer.setScissor(viewport.x, viewport.y, viewport.width, viewport.height)
      renderer.setScissorTest(true)
      this.currentMesh.material = this.currentMaterial
      this.currentMesh.render(renderer)
    }

    renderer.setRenderTarget(this.resolveRenderTarget)
    renderer.setViewport(0, 0, atlasSize.x, atlasSize.y)
    renderer.setScissor(0, 0, atlasSize.x, atlasSize.y)
    renderer.setScissorTest(false)
    this.resolveMesh.material = this.resolveMaterial
    this.resolveMesh.render(renderer)

    renderer.setScissorTest(false)
    restoreRendererState(renderer, this.rendererState)
    renderer.setViewport(viewportScratch)
    renderer.setScissor(scissorScratch)
    renderer.setScissorTest(scissorTest)

    this.swapBuffers()
    for (let index = 0; index < cascadeCount; ++index) {
      this.previousShadowMatrices[index].copy(this.shadowMaps.cascades[index].matrix)
    }
    this.previousFrameValid = true
    this.cacheCameraPose(this.camera.projectionMatrix)
  }

  override setup(builder: NodeBuilder): unknown {
    const atmosphere = getAtmosphereContext(builder)
    const clouds = getCloudsContext(builder)
    this.setContexts(clouds, atmosphere)

    this.currentMaterial.fragmentNode = this.setupFragmentNode(builder)
    this.currentMaterial.needsUpdate = true
    this.resolveMaterial.fragmentNode = this.setupResolveFragmentNode()
    this.resolveMaterial.needsUpdate = true

    return this.textureNode
  }

  private setupResolveFragmentNode(): Node {
    const currentShadowNode = this.currentRenderTargets.getTextureNode('shadow')
    const depthVelocityNode = this.currentRenderTargets.getTextureNode('depthVelocity')

    const fragmentNode = Fn(() => {
      const coord = ivec2(screenCoordinate).toConst()
      const tileSize = ivec2(this.shadowTileSizeNode).toConst()
      const cascadeIndex = coord.x
        .div(tileSize.x)
        .toInt()
        .clamp(0, this.shadowCascadeCountNode.sub(1))
        .toConst()
      const tileOrigin = ivec2(cascadeIndex.mul(tileSize.x), 0).toConst()
      const tileMaxCoord = tileOrigin.add(tileSize).sub(1).toConst()
      const localCoord = coord.sub(tileOrigin).toConst()
      const localUv = vec2(localCoord).add(0.5).mul(this.shadowTileTexelSizeNode).toConst()
      const outputColor = currentShadowNode.load(coord).toVar()

      If(this.resolveHistoryWeightNode.greaterThan(EPSILON), () => {
        const closestCoord = coord.toVar()
        const closestDepth = float(1e9).toVar()

        for (const offset of closestNeighborOffsets) {
          const neighborCoord = coord.add(offset).clamp(tileOrigin, tileMaxCoord).toConst()
          const neighborDepth = depthVelocityNode.load(neighborCoord).r.toConst()
          If(neighborDepth.lessThan(closestDepth), () => {
            closestCoord.assign(neighborCoord)
            closestDepth.assign(neighborDepth)
          })
        }

        const depthVelocity = depthVelocityNode.load(closestCoord).toConst()
        const velocityUv = depthVelocity.gb.mul(this.shadowTileTexelSizeNode).toConst()
        const prevLocalUv = localUv.sub(velocityUv).toConst()
        const insideHistory = prevLocalUv
          .greaterThanEqual(vec2(0))
          .all()
          .and(prevLocalUv.lessThanEqual(vec2(1)).all())
          .toConst()

        If(insideHistory, () => {
          const prevAtlasUv = vec2(
            prevLocalUv.x.add(cascadeIndex.toFloat()).div(this.shadowCascadeCountNode.toFloat()),
            prevLocalUv.y
          ).toConst()
          const historyColor = this.historyNode.sample(prevAtlasUv).toConst()
          const clippedHistory = varianceClippingTile(
            currentShadowNode,
            coord,
            tileOrigin,
            tileMaxCoord,
            outputColor,
            historyColor,
            float(CLOUD_SHADOW_VARIANCE_GAMMA)
          ).toConst()
          outputColor.assign(
            mix(clippedHistory, outputColor, float(CLOUD_SHADOW_TEMPORAL_ALPHA))
          )
        })
      })

      return outputColor
    })()

    return fragmentNode
  }

  private setupFragmentNode(builder: NodeBuilder): Node {
    const atmosphere = getAtmosphereContext(builder)
    const clouds = getCloudsContext(builder)
    const unitToWorld = float(1).div(atmosphere.worldToUnit).toConst()
    const altitudeCorrectionUnit = select(
      clouds.correctAltitudeNode,
      atmosphere.altitudeCorrectionUnit,
      vec3(0)
    ).toConst()

    const sampleExtinctionAt = Fn(
      ([positionUnit, mipLevel, highFrequencyWeight]: [
        Node<'vec3'>,
        Node<'float'>,
        Node<'float'>
      ]) => {
        const height = positionUnit
          .length()
          .sub(atmosphere.bottomRadius)
          .mul(unitToWorld)
          .toConst()
        const extinction = float(0).toVar()

        If(
          height
            .greaterThanEqual(clouds.shadowMinHeightNode)
            .and(height.lessThanEqual(clouds.shadowMaxHeightNode)),
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
              .mul(clouds.shadowLayerMaskNode)
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
            const localWeatherSpeed = clouds.localWeatherOffsetNode.length().toConst()
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
              clouds.turbulenceNode.and(highFrequencyWeight.greaterThan(EPSILON)),
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
                    .mul(clouds.turbulenceDisplacementNode.mul(turbulenceAmount))
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
            const detailFade = highFrequencyWeight
              .mul(densityEdgeFade.pow2())
              .mul(float(1).sub(remapClamp(mipLevel, float(0.1), float(0.85))))
              .toConst()
            const shapeDetailPosition = positionWorld
              .add(turbulence)
              .mul(clouds.shapeDetailRepeatNode)
              .add(clouds.shapeDetailOffsetNode)
              .toConst()
            const detail = this.shapeDetailTextureNode
              .sample(shapeDetailPosition)
              .r.toConst()

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
                    .mul(exp(clouds.densityProfileExponentsNode.mul(heightFraction)))
                    .add(clouds.densityProfileLinearTermsNode.mul(heightFraction))
                    .add(clouds.densityProfileConstantTermsNode)
                )
                .saturate()
            )
            const densitySum = density.x
              .add(density.y)
              .add(density.z)
              .add(density.w)
              .toConst()

            If(densitySum.greaterThan(this.minDensityNode), () => {
              extinction.assign(
                densitySum
                  .mul(
                    clouds.scatteringCoefficientNode.add(
                      clouds.absorptionCoefficientNode
                    )
                  )
                  .max(this.minExtinctionNode)
              )
            })
          }
        )

        return extinction
      }
    )

    const fragmentNode = Fn(() => {
      const clip = uv().mul(2).sub(1).toConst()
      const pointWorld = this.currentInverseShadowMatrixNode
        .mul(vec4(clip, -1, 1))
        .toVar()
      pointWorld.assign(pointWorld.div(pointWorld.w.abs().max(EPSILON)))

      const rayOriginUnit = atmosphere.matrixWorldToECEF
        .mul(vec4(pointWorld.xyz, 1))
        .xyz.mul(atmosphere.worldToUnit)
        .add(altitudeCorrectionUnit)
        .toConst()
      const rayDirectionUnit = atmosphere.sunDirectionECEF
        .negate()
        .normalize()
        .toConst()
      const minRadius = atmosphere.bottomRadius
        .add(clouds.shadowMinHeightNode.mul(atmosphere.worldToUnit))
        .toConst()
      const maxRadius = atmosphere.bottomRadius
        .add(clouds.shadowMaxHeightNode.mul(atmosphere.worldToUnit))
        .toConst()
      const firstIntersections = raySpheresIntersections(
        rayOriginUnit,
        rayDirectionUnit,
        vec3(0),
        vec4(maxRadius, minRadius, atmosphere.bottomRadius, atmosphere.bottomRadius)
      )
        .get('near')
        .toConst()
      const segmentStart = max(firstIntersections.x, 0).toVar()
      const segmentEnd = select(
        firstIntersections.y.lessThan(0),
        float(1e6).mul(atmosphere.worldToUnit),
        firstIntersections.y
      ).toVar()

      const extinctionSum = float(0).toVar()
      const maxOpticalDepth = float(0).toVar()
      const maxOpticalDepthTail = float(0).toVar()
      const transmittanceIntegral = float(1).toVar()
      const weightedDistanceSum = float(0).toVar()
      const transmittanceSum = float(0).toVar()
      const sampleCount = float(0).toVar()

      If(segmentEnd.greaterThan(segmentStart), () => {
        const jitter = float(0.5).toConst()
        const segmentLengthWorld = segmentEnd.sub(segmentStart).mul(unitToWorld).toConst()
        const stepSizeWorld = segmentLengthWorld
          .div(this.maxIterationCountNode.toFloat().max(1))
          .clamp(this.minStepSizeNode, this.maxStepSizeNode)
          .toConst()
        const rayDistance = segmentStart
          .add(stepSizeWorld.mul(atmosphere.worldToUnit).mul(jitter))
          .toVar()

        Loop({ start: 0, end: WEBGPU_MAX_PRIMARY_STEPS, condition: '<' }, ({ i }) => {
          If(float(i).greaterThanEqual(this.maxIterationCountNode.toFloat()), () => {
            Break()
          })
          If(rayDistance.greaterThanEqual(segmentEnd), () => {
            Break()
          })

          const positionUnit = rayOriginUnit
            .add(rayDirectionUnit.mul(rayDistance))
            .toConst()
          const rayDistanceWorld = rayDistance
            .sub(segmentStart)
            .mul(unitToWorld)
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
          const extinction = sampleExtinctionAt(
            positionUnit,
            this.currentShadowMipLevelNode,
            highFrequencyWeight
          ).toConst()

          If(extinction.greaterThan(this.minExtinctionNode), () => {
            const sampleTransmittance = exp(
              extinction.negate().mul(stepSizeWorld)
            ).toConst()
            extinctionSum.addAssign(extinction)
            maxOpticalDepth.addAssign(extinction.mul(stepSizeWorld))
            transmittanceIntegral.mulAssign(sampleTransmittance)
            weightedDistanceSum.addAssign(
              rayDistanceWorld.mul(transmittanceIntegral)
            )
            transmittanceSum.addAssign(transmittanceIntegral)
            sampleCount.addAssign(1)

            If(
              transmittanceIntegral.lessThanEqual(this.minTransmittanceNode),
              () => {
                maxOpticalDepthTail.assign(
                  min(
                    this.opticalDepthTailScaleNode
                      .mul(stepSizeWorld)
                      .mul(exp(sampleCount.oneMinus())),
                    stepSizeWorld.mul(0.5)
                  )
                )
              }
            )

            If(
              transmittanceIntegral.lessThanEqual(this.minTransmittanceNode),
              () => {
              Break()
              }
            )
          })

          rayDistance.addAssign(stepSizeWorld.mul(atmosphere.worldToUnit))
          Continue()
        })
      })

      const shadowData = select(
        sampleCount.equal(0),
        vec4(segmentEnd.sub(segmentStart).mul(unitToWorld), 0, 0, 0),
        vec4(
          min(
            weightedDistanceSum.div(transmittanceSum.max(EPSILON)),
            segmentEnd.sub(segmentStart).mul(unitToWorld)
          ),
          extinctionSum.div(sampleCount.max(EPSILON)),
          maxOpticalDepth.max(0),
          maxOpticalDepthTail.max(0)
        )
      ).toConst()
      const frontPositionUnit = rayOriginUnit
        .add(
          rayDirectionUnit.mul(
            segmentStart.add(shadowData.x.mul(atmosphere.worldToUnit))
          )
        )
        .toConst()
      const frontPositionWorld = atmosphere.matrixECEFToWorld
        .mul(
          vec4(
            frontPositionUnit.sub(altitudeCorrectionUnit).div(atmosphere.worldToUnit),
            1
          )
        )
        .xyz
        .toConst()
      const prevClip = this.currentReprojectionMatrixNode
        .mul(vec4(frontPositionWorld, 1))
        .toConst()
      const prevLocalUv = prevClip.xy
        .div(prevClip.w.abs().max(EPSILON))
        .mul(0.5)
        .add(0.5)
        .toConst()
      const shadowVelocity = uv().sub(prevLocalUv).mul(this.shadowTileSizeNode).toConst()

      return mrt({
        shadow: shadowData,
        depthVelocity: vec4(shadowData.x, shadowVelocity, 0)
      })
    })()

    ;(
      fragmentNode as Node & {
        isOutputStructNode?: boolean
      }
    ).isOutputStructNode = true

    return fragmentNode
  }

  override dispose(): void {
    this.currentRenderTargets.dispose()
    this.resolveRenderTarget.dispose()
    this.historyRenderTarget.dispose()
    this.currentMaterial.dispose()
    this.resolveMaterial.dispose()
    this.currentMesh.geometry.dispose()
    this.resolveMesh.geometry.dispose()
    super.dispose()
  }
}

export const cloudsShadow = (
  ...args: ConstructorParameters<typeof CloudsShadowNode>
): CloudsShadowNode => new CloudsShadowNode(...args)

export interface CloudShadowOptions {
  shadowNode: CloudsShadowNode
  normalNode?: Node<'vec3'> | null
}

export const cloudShadow = (
  positionWorldNode: Node<'vec3'>,
  { shadowNode, normalNode = null }: CloudShadowOptions
): Node<'float'> => shadowNode.sample(positionWorldNode, normalNode)
