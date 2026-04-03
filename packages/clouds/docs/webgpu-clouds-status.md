# WebGPU Clouds Status

Last updated: 2026-03-31

This note records the current status of `@takram/three-clouds/webgpu` after the M1 acceptance pass and the M2 temporal-upscale acceptance pass.

## M1 Acceptance Status

### Accepted

- `@takram/three-clouds/webgpu` is exported from the package root.
- The WebGPU entry exports `CloudsContext`, `CloudsNode`, `clouds(...)`, and `getCloudsContext(...)`.
- The WebGPU path is implemented as a separate node/context pipeline and does not modify the legacy WebGL `CloudsEffect` / `CloudsPass` path.
- The integration path uses `renderer.contextNode = context({ getAtmosphere, getClouds })`.
- The current composition order is `aerialPerspective -> clouds -> toneMapping`.
- The baseline WebGPU path is still the M1 full-resolution single-frame marching path. It does not depend on temporal history.
- The depth path is wired through `depthToViewZ(... logarithmicDepthBuffer)` and the story uses `logarithmicDepthBuffer: true`.
- Core M1 modeling parameters are exposed in the WebGPU context, including:
  - `shapeDetail`
  - `turbulence`
  - `haze`
  - quality preset
  - resolution scale
  - cloud motion / weather motion
- The WebGPU examples exist in `storybook-webgpu` and currently cover:
  - `GroundBaseline`
  - `GroundAnimated`
  - `CruiseView`
- Runtime acceptance has been performed in browser:
  - the WebGL reference story renders correctly
  - the WebGPU story renders correctly
  - the WebGPU canvas was verified to use a real `webgpu` context, not `webgl2`
  - the animated story was verified to change over time
- Type-check and unit tests passed for the current M1 implementation.
- The Storybook WebGL fallback has been restored so unsupported browsers can fall back to WebGL2.

### M1 Conclusion

M1 can be accepted as a phase milestone.

This means the WebGPU cloud path is now usable and testable as a real rendering path, and the original WebGL path remains intact. It does **not** mean the WebGPU result is already visually matched to the WebGL implementation or ready for final release-quality polish.

## M2 Acceptance Status

### Accepted With Noted Limitations

- `CloudsContext` exposes the M2 control surface:
  - `temporalUpscale`
  - `invalidateHistory()`
  - `historyInvalidationRevision`
- History invalidation is wired for:
  - manual `invalidateHistory()`
  - `temporalUpscale` toggle
  - `qualityPreset` change
  - `resolutionScale` change
  - `shapeDetail` / `turbulence` / `haze` toggles
  - texture replacement for weather / shape / detail / turbulence / STBN textures
- `CloudsNode` now implements the M2 rendering path when `temporalUpscale` is enabled.
- The low-resolution marching pass outputs:
  - cloud-composited color
  - packed transmittance/depth data
  - cloud reprojection velocity
  - cloud depth
- A full-resolution resolve-input pass upsamples cloud color, velocity, depth, and mask from the same low-resolution donor coordinate.
- The full-resolution resolve input keeps `output / velocity / depth / mask` on the same low-resolution donor coordinate.
- `TemporalAntialiasNode` is the M2 history / resolve stage.
- The resolve-input pass still produces a per-pixel current-frame mask as an intermediate output.
- The accepted M2 route does **not** bind that mask into `TemporalAntialiasNode`.
  The temporal resolve intentionally runs as `full-res resolve-input + generic TAA` with `currentFrameMaskNode = null`.
- History reset is now handled in the library for:
  - explicit context invalidation
  - camera cuts detected from position / rotation / projection deltas
  - low-resolution march target resize
  - full-resolution resolve target resize
- `CloudsContext` tracks previous weather / shape / shape-detail offsets frame-to-frame.
- Reprojection includes cloud-field motion from:
  - shape offset deltas
  - shape-detail offset deltas
  - local-weather offset deltas projected back to world space through a local globe-UV Jacobian approximation
- Storybook exposes the primary M2 acceptance stories:
  - `GroundBaseline`
  - `GroundTemporalUpscale`
  - `GroundTemporalUpscaleAnimated`
  - `CruiseView` with a default visible cloud layer
- Storybook also exposes M2-specific validation stories:
  - `GroundTemporalUpscaleCameraCut`
  - `GroundTemporalUpscaleResizeReset`
- `resolutionScale` is now documented and treated as the cloud marching resolution scale. It does not change the final resolved output size.
- `ultra` is intentionally kept semantically aligned with `high`; it only increases primary march sampling density and does not introduce different coverage / termination heuristics.

### M2 Conclusion

M2 can be accepted with noted limitations.

This means the WebGPU clouds path now has a stable low-resolution march plus temporal resolve path, with history invalidation and acceptance stories wired for runtime validation. It does **not** mean the WebGPU path is feature-equivalent to the legacy WebGL clouds path, nor that it uses a clouds-specific temporal resolve implementation.

## M2 Temporal Resolve Decision Record

The project evaluated two M2 resolve directions:

- `mask-feeding generic TAA`
- `full-res resolve-input + generic TAA`

The accepted M2 decision is:

- keep `full-res resolve-input + generic TAA` as the source of truth
- keep generating `mask` as an intermediate output for analysis and possible future work
- do **not** feed that `mask` into `TemporalAntialiasNode` in the accepted M2 path

Why this decision was taken:

- the generic TAA path proved more stable under real browser validation
- it is easier to reason about camera cut, resize, and history reset behavior
- it preserves the standard generic TAA rejection / clipping path
- the masked path is closer to legacy WebGL resolve semantics, but was more fragile under the current donor / velocity / depth assumptions

What this means for future work:

- M2 should continue improving donor quality, velocity/depth consistency, and generic TAA tuning
- if future work wants to move closer to legacy WebGL resolve behavior, that should be treated as a deliberate follow-up investigation rather than an implicit M2 assumption
- the likely long-term end state for WebGL-level parity is still a clouds-specific temporal resolve, not an indefinitely patched generic TAA masked branch

## M2 Regression Note: Story-Layer History Resets

The project hit a temporary M2 regression where `GroundTemporalUpscale` looked much more blocky than the accepted `full-res resolve-input + generic TAA` route should have looked.

Root cause:

- `storybook-webgpu/src/hooks/useTransientControl.ts` used to invoke the initial `onChange(value)` directly during render
- `storybook-webgpu/src/clouds/Clouds-Story.tsx` called `applyCloudStoryPreset(...)` inside that control callback
- `storybook-webgpu/src/clouds/storyPresets.ts` used to call `cloudsContext.invalidateHistory()` unconditionally

Effect:

- React re-renders kept invalidating temporal history
- the M2 path could not converge
- the result looked like persistent low-resolution donor blocks instead of accumulated temporal resolve

Fix:

- `useTransientControl(...)` now applies its initial callback in a layout effect instead of render
- the clouds story only reapplies the preset when `viewPreset` actually changes
- `applyCloudStoryPreset(...)` only invalidates history when preset values actually change

Conclusion:

- the abnormal mosaic regression was not caused by the accepted M2 runtime route itself
- it was caused by story-layer history resets defeating temporal accumulation

## Explicitly Out Of Scope For M1

The following items are not part of M1 and should not be used to block M1 acceptance:

- `temporalUpscale`
- TAA / temporal history resolve
- `shadow.*`
- `lightShafts`
- WebGPU `ProceduralTexture` / `Procedural3DTexture`
- `/webgpu/r3f` convenience entry
- standalone clouds without `AtmosphereContext`
- global / orbital coverage model upgrades

## M2 Remaining Limitations

- The temporal resolve stage is still the generic `TemporalAntialiasNode`.
  This is intentional for M2 and means the WebGPU path does not yet match the legacy WebGL quarter-resolution clouds resolve logic one-to-one.
- The current-frame `mask` is still generated by resolve-input, but the accepted M2 path does not consume it in the final TAA stage.
  The current acceptance target remains stable donor sampling, history reset, motion, and default story visibility rather than WebGL-equivalent resolve behavior.
- M2 still does not include:
  - `shadow.*`
  - `lightShafts`
  - WebGPU `ProceduralTexture` / `Procedural3DTexture`
  - `/webgpu/r3f`
- Real acceptance still requires a desktop browser session with actual WebGPU enabled.
  Headless screenshots and WebGL fallback output remain diagnostic-only and are not the source of truth.
- Visual parity with the WebGL clouds path is not guaranteed.
  M2 targets stable motion, reset behavior, and materially better output than raw quarter-resolution upsampling, not pixel-identical matching.

## Notes For Future Work

- Do not back-port WebGPU logic into the legacy WebGL `CloudsEffect`.
- Keep WebGPU procedural textures out of the public API until they are implemented.
- Keep M2 focused on temporal upscaling and resolve quality. Do not mix M3 scope into the same change set.

## M3 Cloud Shadows Status

### Accepted With Parity Notes

- WebGPU now has a dedicated `CloudsShadowNode` atlas path.
- `CloudsContext.shadow` is wired and drives:
  - cascade count
  - atlas map size
  - split configuration
  - Beer-shadow marching quality parameters
- `cloudShadow(positionWorld, { shadowNode, normalNode })` is available for explicit receiver materials.
- `CloudsShadowLengthNode` is available and can be injected through
  `renderer.contextNode.getCloudsShadowLength()` for atmosphere-side shafts.
- `AerialPerspectiveNode` now supports automatic cloud-shadow and shadow-length
  provider hooks through `getCloudsShadow()` and `getCloudsShadowLength()`.
  This enables global scene-side attenuation and shafts without per-material
  manual wiring.
- `storybook-webgpu` includes:
  - `GroundCloudShadows`
  - `GroundCloudShadowsAnimated`
  - `GroundCloudShadowsDebugAtlas`
  - `GroundLightShaftsGlobal`
  - `GroundLightShaftsCameraCut`
  - `GroundLightShaftsResizeReset`
  - `WebGLvsWebGPUParity`
- Real Chrome WebGPU validation has confirmed:
  - the atlas path renders structured cloud coverage
  - the ground receiver path now projects visible cloud shadows
  - the animated ground-shadow story moves continuously instead of staying static
  - global atmospheric shafts respond to scene depth when cloud shadow-length
    provider is enabled

### Important Fixes Landed During M3

- `CloudsShadowLengthNode` no longer behaves as a single-frame scalar buffer only.
  It now has a dedicated WebGPU `current -> resolve -> history` path:
  - current pass outputs full-resolution `shadowLength + depthVelocity`
  - resolve pass reprojects history with nearest-depth donor selection
  - variance clipping and temporal mix now follow the legacy WebGL defaults
    (`temporalAlpha = 0.1`, `varianceGamma = 2`)
- `CloudsShadowLengthNode` now resets history on the same practical triggers
  used by the rest of the WebGPU clouds path:
  - explicit `CloudsContext.invalidateHistory()`
  - render-size changes
  - camera cuts detected from position / rotation / projection deltas
- Shadow-length reprojection is now based on full-resolution scene/far
  front-depth velocity instead of a single-frame buffer with no temporal
  stabilization.
- The shadow-length path no longer hardcodes its internal step-growth factor.
  It now follows `clouds.perspectiveStepScale`, and the parity story preset pins
  that value to `1.01` when comparing against WebGL.
- Cascade selection on the receiver side now follows the same normalized-depth semantics as the shared/WebGL CSM helper.
  The broken path compared raw `viewDepth` against normalized split intervals.
- Receiver sampling no longer rejects samples only because projected `ndc.z` falls outside `[-1, 1]`.
  For the current Beer shadow atlas, UV coverage is the meaningful validity test for the receiver lookup.
- Receiver-side cascade transitions now use boundary-based blend weights plus valid-sample renormalization instead of the earlier wide interval-branch blend.
  This removed the obvious horizontal cascade break bands that were visible on the ground receiver story.
- Receiver-side atlas reads now use a small tent-style 9-tap filter instead of the earlier 5-tap cross.
  This is still single-frame filtering, but it reduces the harshest jagged edges on ground shadows without introducing a new temporal path.
- The shadow story is back on a normal receiver-material presentation instead of the temporary high-gain debug visualization that was used during bring-up.
- The WebGPU cloud shadow atlas is no longer treated as a single-channel receiver-only transmittance map.
  The atlas now packs Beer-shadow-style data closer to the legacy WebGL path:
  - transmittance-weighted front depth
  - mean extinction
  - accumulated optical depth
  - optical-depth tail
- `CloudsNode` can now optionally feed that packed atlas back into the cloud-body lighting path.
  Instead of multiplying sun light by a receiver-style shadow factor, the WebGPU path now reconstructs Beer optical depth from the atlas and adds it to the cloud sample's sun-ray optical depth, which is closer to how the legacy WebGL `clouds.frag` path combines local sun marching with the Beer shadow map.
- The cloud-body Beer shadow lookup is no longer limited to the receiver-oriented filtered atlas path.
  WebGPU now uses a dedicated cloud-body sampling path with:
  - a single selected cascade instead of the receiver blend path
  - a horizon-weighted filter radius matching the legacy WebGL `maxShadowFilterRadius` behavior
  - stable per-pixel PCF rotation for the cloud-body lookup
- Receiver sampling and cloud-body sampling are now intentionally separated.
  This keeps the accepted ground receiver filtering intact while allowing the cloud-body lookup to move closer to the legacy WebGL Beer-shadow semantics.
- The WebGPU cloud-shadow stories that sample `cloudShadow(...)` directly now set the required `AtmosphereContext` / `CloudsContext` on the shared `CloudsShadowNode` up front instead of relying on the screen-space clouds node to do it implicitly.
- The temporary M3 debug path that replaced cloud lighting with direct atlas-visibility output has been removed.
  `CloudsNode` is back on the normal cloud-lighting path, with Beer optical depth folded into the sun-ray optical depth instead of overriding the final lighting with a debug visualization.
- The clouds Beer-shadow validation stories were retuned so they are usable for runtime inspection:
  - the cloud-shadow validation preset now uses denser, shadow-friendly layer parameters similar to the dedicated ground-shadow bring-up story
  - the low-sun validation story no longer uses a near-night timestamp for Fuji; it now targets a golden-hour range so cloud-body attenuation can be inspected without the whole sky collapsing into an almost black scene
- The WebGPU shadow atlas `frontDepth` packing has been realigned to match the WebGL interpretation:
  - atlas `shadow.r` is now accumulated relative to the cloud-top entry segment, not absolute ray distance from the temporary light-space probe point
  - front-position reprojection in `CloudsShadowNode` now reconstructs world position using `segmentStart + frontDepth`
- `GroundBeerShadows*` stories no longer change cloud coverage / cloud-layer densities when toggling `cloudShadowAtlas`.
  The toggle now isolates shadow-path behavior instead of coupling feature enablement with volume-shape retuning.
- `CloudsShadowLengthNode` marching semantics were aligned closer to WebGL `marchShadowLength` without forking a separate public API.
  The runtime now stays on the normal context-driven step controls, and the
  parity story preset pins the relevant values for comparison.
- Light-shaft validation stories now use shadow-length marching budgets that are explicit in story presets (`maxShadowLengthIterationCount=500`, `minShadowLengthStepSize=50`, `maxShadowLengthRayDistance=2e5`) so acceptance visibility is not coupled to unrelated presets.

### Remaining Limitations

- WebGPU shadows are still single-frame atlas based and do not yet use the
  legacy WebGL temporal shadow resolve pass.
- Visual parity is high-level semantic parity, not pixel parity. Daylight
  cloud-body attenuation can still differ from WebGL in edge cases.
- The following remain out of scope:
  - WebGPU procedural textures for clouds
  - `/webgpu/r3f` convenience wrappers
