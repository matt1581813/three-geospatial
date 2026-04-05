import {
  HalfFloatType,
  LinearFilter,
  Matrix4,
  NearestFilter,
  Quaternion,
  RenderTarget,
  Vector2,
  Vector3,
  Vector4,
  type Camera,
  type Texture
} from 'three'
import { hash } from 'three/src/nodes/core/NodeUtils.js'
import {
  Break,
  exp,
  float,
  Fn,
  If,
  ivec2,
  Loop,
  max,
  mix,
  min,
  mrt,
  remapClamp,
  screenCoordinate,
  select,
  sqrt,
  texture,
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
  cameraFar,
  cameraNear,
  depthToViewZ,
  inverseProjectionMatrix,
  inverseViewMatrix,
  outputTexture,
  projectionMatrix,
  raySphereIntersection,
  screenToPositionView,
  type Node
} from '@takram/three-geospatial/webgpu'

import { CloudsContext, getCloudsContext } from './CloudsContext'
import { CloudsRenderTargets } from './CloudsRenderTargets'
import { CloudsShadowNode } from './CloudsShadowNode'
import { CloudsTemporalState } from './CloudsTemporalState'

const { resetRendererState, restoreRendererState } = RendererUtils

const sizeScratch = /*#__PURE__*/ new Vector2()
const viewportScratch = /*#__PURE__*/ new Vector4()
const scissorScratch = /*#__PURE__*/ new Vector4()
const viewProjectionMatrixScratch = /*#__PURE__*/ new Matrix4()
const viewReprojectionMatrixScratch = /*#__PURE__*/ new Matrix4()
const cameraPositionScratch = /*#__PURE__*/ new Vector3()
const cameraQuaternionScratch = /*#__PURE__*/ new Quaternion()
const EPSILON = 1e-6
const WEBGPU_MAX_SHADOW_LENGTH_STEPS = 512
const SHADOW_LENGTH_RESOLUTION_SCALE = 0.25
const SHADOW_LENGTH_CAMERA_CUT_POSITION_THRESHOLD = 1_000
const SHADOW_LENGTH_CAMERA_CUT_ROTATION_THRESHOLD = Math.PI / 12
const SHADOW_LENGTH_CAMERA_CUT_PROJECTION_THRESHOLD = 1e-3

export const SHADOW_LENGTH_TEMPORAL_ALPHA = 0.1
export const SHADOW_LENGTH_VARIANCE_GAMMA = 2

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

type CloudsShadowLengthCurrentTextureName = 'shadowLength' | 'depthVelocity'

export function clampShadowLengthDistance(
  sceneDistance: number,
  maxDistance: number
): number {
  return Math.max(Math.min(sceneDistance, maxDistance), 0)
}

export function computeShadowLengthContribution(
  opticalDepth: number,
  stepSize: number,
  attenuation = 1
): number {
  return (
    (1 - Math.exp(-Math.max(opticalDepth, 0))) *
    Math.max(stepSize, 0) *
    Math.max(attenuation, 0)
  )
}

function getMaxMatrixDelta(a: Matrix4, b: Matrix4): number {
  let delta = 0
  for (let index = 0; index < 16; ++index) {
    delta = Math.max(delta, Math.abs(a.elements[index] - b.elements[index]))
  }
  return delta
}

const clipAABB = /*#__PURE__*/ FnLayout({
  name: 'clipShadowLengthAABB',
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

const varianceClippingFullscreen = /*#__PURE__*/ FnVar(
  (
    inputNode: TextureNode,
    coord: Node<'ivec2'>,
    maxCoord: Node<'ivec2'>,
    current: Node<'vec4'>,
    history: Node<'vec4'>,
    gamma: Node<'float'>
  ): Node<'vec4'> => {
    const moment1 = current.toVar()
    const moment2 = current.pow2().toVar()

    for (const offset of varianceOffsets) {
      const neighborCoord = coord.add(offset).clamp(ivec2(0), maxCoord).toConst()
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

export class CloudsShadowLengthNode extends TempNode {
  static override get type(): string {
    return 'CloudsShadowLengthNode'
  }

  readonly depthNode: TextureNode
  readonly camera: Camera
  readonly shadowNode: CloudsShadowNode

  private readonly currentRenderTargets =
    new CloudsRenderTargets<CloudsShadowLengthCurrentTextureName>(null, {
      colorAttachments: ['shadowLength', 'depthVelocity']
    })
  private resolveRenderTarget = this.createRenderTarget('Resolve')
  private historyRenderTarget = this.createRenderTarget('History')
  private readonly textureNode = outputTexture(this, this.resolveRenderTarget.texture)
  private readonly currentMaterial = new NodeMaterial()
  private readonly resolveMaterial = new NodeMaterial()
  private readonly currentMesh = new QuadMesh(this.currentMaterial)
  private readonly resolveMesh = new QuadMesh(this.resolveMaterial)
  private readonly historyNode = texture(this.historyRenderTarget.texture)
  private rendererState?: RendererUtils.RendererState
  private cloudsContext: CloudsContext | null = null
  private atmosphereContext: AtmosphereContext | null = null
  private readonly temporalState = new CloudsTemporalState()
  private readonly previousViewProjectionMatrix = new Matrix4()
  private readonly previousCameraPosition = new Vector3()
  private readonly previousCameraQuaternion = new Quaternion()
  private readonly previousProjectionMatrix = new Matrix4()
  private previousFrameValid = false
  private cameraPoseValid = false

  private readonly viewReprojectionNode = uniform(new Matrix4()).setName(
    'cloudsShadowLengthViewReprojection'
  )

  private readonly outputSizeNode = uniform(new Vector2(1, 1)).setName(
    'cloudsShadowLengthOutputSize'
  )

  private readonly resolveHistoryWeightNode = uniform(0).setName(
    'cloudsShadowLengthResolveHistoryWeight'
  )

  constructor(
    depthNode: TextureNode,
    camera: Camera,
    shadowNode: CloudsShadowNode
  ) {
    super('float')
    this.depthNode = depthNode
    this.camera = camera
    this.shadowNode = shadowNode
    this.currentRenderTargets.getTexture('depthVelocity').minFilter = NearestFilter
    this.currentRenderTargets.getTexture('depthVelocity').magFilter = NearestFilter
    this.currentMaterial.name = 'CloudsShadowLengthNode.Current'
    this.resolveMaterial.name = 'CloudsShadowLengthNode.Resolve'
    this.updateBeforeType = NodeUpdateType.FRAME
  }

  override customCacheKey(): number {
    return hash(this.camera.id)
  }

  private createRenderTarget(name: string): RenderTarget {
    const renderTarget = new RenderTarget(1, 1, {
      depthBuffer: false,
      type: HalfFloatType
    })
    renderTarget.texture.name = `CloudsShadowLengthNode.${name}`
    renderTarget.texture.minFilter = LinearFilter
    renderTarget.texture.magFilter = LinearFilter
    renderTarget.texture.generateMipmaps = false
    return renderTarget
  }

  setContexts(cloudsContext: CloudsContext, atmosphereContext: AtmosphereContext): void {
    this.cloudsContext = cloudsContext
    this.atmosphereContext = atmosphereContext
    this.shadowNode.setContexts(cloudsContext, atmosphereContext)
  }

  getTexture(): Texture {
    return this.textureNode.value
  }

  getTextureNode(): TextureNode {
    return this.textureNode
  }

  sampleShadowLength(uvNode: Node<'vec2'> = uv()): Node<'float'> {
    return this.textureNode.sample(uvNode).r
  }

  private setSize(width: number, height: number): boolean {
    if (
      this.currentRenderTargets.renderTarget.width === width &&
      this.currentRenderTargets.renderTarget.height === height
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

  private detectCameraCut(projectionMatrixValue: Matrix4): boolean {
    if (!this.cameraPoseValid) {
      return false
    }

    cameraPositionScratch.setFromMatrixPosition(this.camera.matrixWorld)
    cameraQuaternionScratch.setFromRotationMatrix(this.camera.matrixWorld)

    const positionDelta = cameraPositionScratch.distanceTo(this.previousCameraPosition)
    const rotationDelta =
      2 *
      Math.acos(
        Math.min(
          1,
          Math.abs(cameraQuaternionScratch.dot(this.previousCameraQuaternion))
        )
      )
    const projectionDelta = getMaxMatrixDelta(
      projectionMatrixValue,
      this.previousProjectionMatrix
    )

    return (
      positionDelta > SHADOW_LENGTH_CAMERA_CUT_POSITION_THRESHOLD ||
      rotationDelta > SHADOW_LENGTH_CAMERA_CUT_ROTATION_THRESHOLD ||
      projectionDelta > SHADOW_LENGTH_CAMERA_CUT_PROJECTION_THRESHOLD
    )
  }

  private cacheCameraPose(projectionMatrixValue: Matrix4): void {
    this.previousCameraPosition.setFromMatrixPosition(this.camera.matrixWorld)
    this.previousCameraQuaternion.setFromRotationMatrix(this.camera.matrixWorld)
    this.previousProjectionMatrix.copy(projectionMatrixValue)
    this.cameraPoseValid = true
  }

  override updateBefore({ renderer, deltaTime, frameId }: NodeFrame): void {
    if (renderer == null) {
      return
    }
    const clouds = this.cloudsContext
    const atmosphere = this.atmosphereContext
    if (clouds == null || atmosphere == null) {
      return
    }

    clouds.advance(frameId, deltaTime)
    this.temporalState.observe(clouds)
    this.shadowNode.setContexts(clouds, atmosphere)

    const size = renderer.getDrawingBufferSize(sizeScratch)
    const width = Math.max(
      Math.round(size.x * SHADOW_LENGTH_RESOLUTION_SCALE),
      1
    )
    const height = Math.max(
      Math.round(size.y * SHADOW_LENGTH_RESOLUTION_SCALE),
      1
    )
    const resized = this.setSize(width, height)
    this.outputSizeNode.value.set(width, height)

    viewProjectionMatrixScratch.multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse
    )
    const useHistory =
      this.previousFrameValid &&
      !resized &&
      !this.temporalState.consumeHistoryReset() &&
      !this.detectCameraCut(this.camera.projectionMatrix)
    this.resolveHistoryWeightNode.value = useHistory ? 1 : 0

    viewReprojectionMatrixScratch.multiplyMatrices(
      useHistory ? this.previousViewProjectionMatrix : viewProjectionMatrixScratch,
      this.camera.matrixWorld
    )
    this.viewReprojectionNode.value.copy(viewReprojectionMatrixScratch)

    renderer.getViewport(viewportScratch)
    renderer.getScissor(scissorScratch)
    const scissorTest = renderer.getScissorTest()

    this.rendererState = resetRendererState(renderer, this.rendererState)

    renderer.setRenderTarget(this.currentRenderTargets.renderTarget)
    renderer.setViewport(0, 0, width, height)
    renderer.setScissor(0, 0, width, height)
    renderer.setScissorTest(false)
    renderer.setClearColor(0x000000, 1)
    renderer.clear()
    this.currentMesh.material = this.currentMaterial
    this.currentMesh.render(renderer)

    renderer.setRenderTarget(this.resolveRenderTarget)
    renderer.setViewport(0, 0, width, height)
    renderer.setScissor(0, 0, width, height)
    renderer.setScissorTest(false)
    this.resolveMesh.material = this.resolveMaterial
    this.resolveMesh.render(renderer)

    restoreRendererState(renderer, this.rendererState)
    renderer.setViewport(viewportScratch)
    renderer.setScissor(scissorScratch)
    renderer.setScissorTest(scissorTest)

    this.swapBuffers()
    this.previousViewProjectionMatrix.copy(viewProjectionMatrixScratch)
    this.previousFrameValid = true
    this.cacheCameraPose(this.camera.projectionMatrix)
  }

  override setup(builder: NodeBuilder): unknown {
    const atmosphereContext = getAtmosphereContext(builder)
    const cloudsContext = getCloudsContext(builder)
    this.setContexts(cloudsContext, atmosphereContext)
    this.currentMaterial.fragmentNode = this.setupCurrentFragmentNode(builder)
    this.currentMaterial.needsUpdate = true
    this.resolveMaterial.fragmentNode = this.setupResolveFragmentNode()
    this.resolveMaterial.needsUpdate = true
    return this.sampleShadowLength()
  }

  private setupCurrentFragmentNode(builder: NodeBuilder): Node {
    const clouds = this.cloudsContext
    const atmosphere = this.atmosphereContext
    if (clouds == null || atmosphere == null) {
      return vec4(0)
    }
    const camera = this.camera
    const near = cameraNear(camera)
    const far = cameraFar(camera)
    const logarithmic = builder.renderer.logarithmicDepthBuffer
    const perspective = camera.isPerspectiveCamera
    const unitToWorld = float(1).div(atmosphere.worldToUnit).toConst()

    const fragmentNode = Fn(() => {
      const depth = this.depthNode.sample(uv()).r.toConst()
      const farViewZ = depthToViewZ(float(1), near, far, {
        perspective,
        logarithmic
      }).toConst()
      const farPositionView = screenToPositionView(
        uv(),
        float(1),
        farViewZ,
        projectionMatrix(camera),
        inverseProjectionMatrix(camera)
      ).toConst()
      const sceneViewZ = depthToViewZ(depth, near, far, {
        perspective,
        logarithmic
      }).toConst()
      const scenePositionView = screenToPositionView(
        uv(),
        depth,
        sceneViewZ,
        projectionMatrix(camera),
        inverseProjectionMatrix(camera)
      ).toConst()
      const matrixWorld = inverseViewMatrix(camera).toConst()
      const farDirectionWorld = matrixWorld
        .mul(vec4(farPositionView.normalize(), 0))
        .xyz.normalize()
        .toConst()
      const frontPositionView = mix(
        farPositionView,
        scenePositionView,
        depth.lessThan(float(1)).toFloat()
      ).toConst()
      const frontDepth = select(
        depth.lessThan(float(1)),
        sceneViewZ.negate().max(0),
        float(far)
      ).toConst()
      const prevClip = this.viewReprojectionNode
        .mul(vec4(frontPositionView, 1))
        .toConst()
      const prevUv = prevClip.xy
        .div(prevClip.w.abs().max(EPSILON))
        .mul(0.5)
        .add(0.5)
        .toConst()
      const velocity = uv().sub(prevUv).toConst()
      const outputShadowLength = float(0).toVar()

      If(
        clouds.lightShaftsNode.and(
          clouds.shadowMaxHeightNode.greaterThan(clouds.shadowMinHeightNode)
        ),
        () => {
          const scenePositionWorld = matrixWorld
            .mul(vec4(scenePositionView, 1))
            .xyz.toConst()
          const cameraWorld = matrixWorld
            .mul(vec4(vec3(0), 1))
            .xyz.toConst()
          const sceneDepthBlend = float(1)
            .sub(remapClamp(depth, float(0.999), float(1)))
            .toConst()
          const rayDirectionWorld = mix(
            farDirectionWorld,
            scenePositionWorld.sub(cameraWorld).normalize(),
            sceneDepthBlend
          )
            .normalize()
            .toConst()
          const altitudeCorrectionUnit = select(
            clouds.correctAltitudeNode,
            atmosphere.altitudeCorrectionUnit,
            vec3(0)
          ).toConst()
          const cameraPositionUnit = atmosphere.cameraPositionUnit
            .add(altitudeCorrectionUnit)
            .toConst()
          const scenePositionUnit = atmosphere.matrixWorldToECEF
            .mul(vec4(scenePositionWorld, 1))
            .xyz.mul(atmosphere.worldToUnit)
            .add(altitudeCorrectionUnit)
            .toConst()
          const rayDirectionUnit = atmosphere.matrixWorldToECEF
            .mul(vec4(rayDirectionWorld, 0))
            .xyz.normalize()
            .toConst()
          const sceneDistance = mix(
            clouds.maxShadowLengthRayDistanceNode.mul(atmosphere.worldToUnit),
            max(scenePositionUnit.sub(cameraPositionUnit).dot(rayDirectionUnit), 0),
            sceneDepthBlend
          ).toConst()
          const shadowTopRadius = atmosphere.bottomRadius
            .add(clouds.shadowMaxHeightNode.mul(atmosphere.worldToUnit))
            .toConst()
          const groundIntersections = raySphereIntersection(
            cameraPositionUnit,
            rayDirectionUnit,
            vec3(0),
            atmosphere.bottomRadius
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
          const segmentStart = float(-1).toVar()
          const segmentEnd = float(-1).toVar()

          If(atmosphere.cameraHeight.lessThan(clouds.shadowMaxHeightNode), () => {
            segmentStart.assign(float(near).mul(atmosphere.worldToUnit))
            segmentEnd.assign(
              select(groundHit, groundIntersections.x, shadowTopIntersections.y)
            )
          }).Else(() => {
            segmentStart.assign(max(shadowTopIntersections.x, float(0)))
            segmentEnd.assign(
              select(
                groundHit,
                min(shadowTopIntersections.y, groundIntersections.x),
                shadowTopIntersections.y
              )
            )
          })
          segmentEnd.assign(min(segmentEnd, sceneDistance))

          If(segmentEnd.greaterThan(segmentStart), () => {
            const rayOriginWorld = cameraWorld
              .add(rayDirectionWorld.mul(segmentStart.mul(unitToWorld)))
              .toConst()
            const maxRayDistanceWorld = segmentEnd
              .sub(segmentStart)
              .mul(unitToWorld)
              .max(0)
              .toConst()
            const stepSizeWorld = clouds.minShadowLengthStepSizeNode.toVar()
            const rayDistanceWorld = stepSizeWorld.mul(0.5).toVar()

            Loop(
              { start: 0, end: WEBGPU_MAX_SHADOW_LENGTH_STEPS, condition: '<' },
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
                const opticalDepth = this.shadowNode
                  .sampleOpticalDepth(samplePositionWorld)
                  .toConst()
                outputShadowLength.addAssign(
                  float(1)
                    .sub(exp(opticalDepth.negate()))
                    .mul(stepSizeWorld)
                )
                stepSizeWorld.mulAssign(clouds.perspectiveStepScaleNode)
                rayDistanceWorld.addAssign(stepSizeWorld)
              }
            )
          })
        }
      )

      return mrt({
        shadowLength: vec4(outputShadowLength.mul(atmosphere.worldToUnit), 0, 0, 1),
        depthVelocity: vec4(frontDepth, velocity, 0)
      })
    })()

    ;(
      fragmentNode as Node & {
        isOutputStructNode?: boolean
      }
    ).isOutputStructNode = true

    return fragmentNode
  }

  private setupResolveFragmentNode(): Node {
    const currentShadowLengthNode =
      this.currentRenderTargets.getTextureNode('shadowLength')
    const depthVelocityNode = this.currentRenderTargets.getTextureNode('depthVelocity')

    return Fn(() => {
      const coord = ivec2(screenCoordinate).toConst()
      const outputColor = currentShadowLengthNode.load(coord).toVar()

      If(this.resolveHistoryWeightNode.greaterThan(EPSILON), () => {
        const maxCoord = ivec2(this.outputSizeNode).sub(1).toConst()
        const closestCoord = coord.toVar()
        const closestDepth = float(1e9).toVar()

        for (const offset of closestNeighborOffsets) {
          const neighborCoord = coord.add(offset).clamp(ivec2(0), maxCoord).toConst()
          const neighborDepth = depthVelocityNode.load(neighborCoord).r.toConst()
          If(neighborDepth.lessThan(closestDepth), () => {
            closestCoord.assign(neighborCoord)
            closestDepth.assign(neighborDepth)
          })
        }

        const depthVelocity = depthVelocityNode.load(closestCoord).toConst()
        const velocityUv = depthVelocity.gb.div(this.outputSizeNode).toConst()
        const prevUv = uv().sub(velocityUv).toConst()
        const insideHistory = prevUv
          .greaterThanEqual(vec2(0))
          .all()
          .and(prevUv.lessThanEqual(vec2(1)).all())
          .toConst()

        If(insideHistory, () => {
          const historyColor = this.historyNode.sample(prevUv).toConst()
          const clippedHistory = varianceClippingFullscreen(
            currentShadowLengthNode,
            coord,
            maxCoord,
            outputColor,
            historyColor,
            float(SHADOW_LENGTH_VARIANCE_GAMMA)
          ).toConst()
          outputColor.assign(
            mix(clippedHistory, outputColor, float(SHADOW_LENGTH_TEMPORAL_ALPHA))
          )
        })
      })

      return outputColor
    })()
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

export const cloudsShadowLength = (
  ...args: ConstructorParameters<typeof CloudsShadowLengthNode>
): CloudsShadowLengthNode => new CloudsShadowLengthNode(...args)
