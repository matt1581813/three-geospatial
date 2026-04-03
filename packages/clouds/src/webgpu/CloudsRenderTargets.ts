import {
  DepthTexture,
  HalfFloatType,
  LinearFilter,
  RenderTarget,
  RGBAFormat,
  type Texture
} from 'three'
import { texture } from 'three/tsl'
import type { Node, TextureNode } from 'three/webgpu'

import { outputTexture } from '@takram/three-geospatial/webgpu'

interface CloudsRenderTargetsOptions<Name extends string> {
  colorAttachments: readonly Name[]
  depthAttachment?: Name
}

export class CloudsRenderTargets<Name extends string> {
  readonly renderTarget: RenderTarget
  readonly textures: Record<Name, Texture>
  readonly textureNodes: Record<Name, TextureNode>

  constructor(
    owner: Node | null,
    { colorAttachments, depthAttachment }: CloudsRenderTargetsOptions<Name>
  ) {
    this.renderTarget = new RenderTarget(1, 1, {
      depthBuffer: depthAttachment != null,
      type: HalfFloatType,
      format: RGBAFormat
    })
    this.renderTarget.texture.minFilter = LinearFilter
    this.renderTarget.texture.magFilter = LinearFilter
    this.renderTarget.texture.generateMipmaps = false

    const [primaryAttachment, ...extraColorAttachments] = colorAttachments
    if (primaryAttachment == null) {
      throw new Error('CloudsRenderTargets requires at least one color target.')
    }
    this.renderTarget.texture.name = primaryAttachment

    const textures = {
      [primaryAttachment]: this.renderTarget.texture
    } as Record<Name, Texture>

    for (const attachment of extraColorAttachments) {
      const texture = this.renderTarget.texture.clone()
      texture.isRenderTargetTexture = true
      texture.name = attachment
      this.renderTarget.textures.push(texture)
      textures[attachment] = texture
    }

    if (depthAttachment != null) {
      const depthTexture = new DepthTexture(1, 1)
      depthTexture.isRenderTargetTexture = true
      depthTexture.name = depthAttachment
      this.renderTarget.depthTexture = depthTexture
      textures[depthAttachment] = depthTexture
    }

    this.textures = textures

    const textureNodes = {} as Record<Name, TextureNode>
    for (const name of [...colorAttachments, depthAttachment].filter(
      (value): value is Name => value != null
    )) {
      textureNodes[name] =
        owner != null ? outputTexture(owner, textures[name]) : texture(textures[name])
    }
    this.textureNodes = textureNodes
  }

  setSize(width: number, height: number): void {
    this.renderTarget.setSize(width, height)
  }

  getTexture(name: Name): Texture {
    return this.textures[name]
  }

  getTextureNode(name: Name): TextureNode {
    return this.textureNodes[name]
  }

  dispose(): void {
    this.renderTarget.dispose()
  }
}
