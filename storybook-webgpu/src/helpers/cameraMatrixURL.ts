import { Matrix4, Vector3, type Camera } from 'three'

const CAMERA_MATRIX_QUERY_KEY = 'cameraMatrix'
const CAMERA_QUERY_KEY = 'camera'
const CAMERA_MATRIX_ELEMENT_COUNT = 16
const CAMERA_COMPONENT_COUNT = 6
const WORLD_UP = new Vector3(0, 1, 0)

function getRelatedWindows(): Window[] {
  if (typeof window === 'undefined') {
    return []
  }

  const related: Window[] = [window]
  try {
    if (window.parent != null && window.parent !== window) {
      const _ = window.parent.location.href
      related.push(window.parent)
    }
  } catch {
    // Ignore cross-origin parent windows.
  }

  return related
}

function readParamFromWindow(target: Window): string | null {
  try {
    const url = new URL(target.location.href)
    return (
      url.searchParams.get(CAMERA_QUERY_KEY) ??
      url.searchParams.get(CAMERA_MATRIX_QUERY_KEY)
    )
  } catch {
    return null
  }
}

export function serializeCameraMatrixElements(
  elements: readonly number[],
  fractionDigits = 6
): string {
  return Array.from(elements)
    .slice(0, CAMERA_MATRIX_ELEMENT_COUNT)
    .map(value =>
      Number.isFinite(value) ? value.toFixed(fractionDigits) : '0'
    )
    .join(',')
}

export function parseCameraMatrixElements(
  serialized: string | null | undefined
): number[] | null {
  if (serialized == null || serialized === '') {
    return null
  }
  const parsed = serialized
    .split(',')
    .map(value => Number.parseFloat(value.trim()))
  if (
    parsed.length !== CAMERA_MATRIX_ELEMENT_COUNT ||
    parsed.some(value => !Number.isFinite(value))
  ) {
    return null
  }
  return parsed
}

export function serializeCameraComponents(
  components: readonly number[],
  fractionDigits = 6
): string {
  return Array.from(components)
    .slice(0, CAMERA_COMPONENT_COUNT)
    .map(value =>
      Number.isFinite(value) ? value.toFixed(fractionDigits) : '0'
    )
    .join(',')
}

export function parseCameraComponents(
  serialized: string | null | undefined
): number[] | null {
  if (serialized == null || serialized === '') {
    return null
  }
  const parsed = serialized
    .split(',')
    .map(value => Number.parseFloat(value.trim()))
  if (
    parsed.length !== CAMERA_COMPONENT_COUNT ||
    parsed.some(value => !Number.isFinite(value))
  ) {
    return null
  }
  return parsed
}

export function readCameraFromUrl(): number[] | null {
  const related = getRelatedWindows()
  for (const target of related) {
    const parsed = parseCameraComponents(readParamFromWindow(target))
    if (parsed != null) {
      return parsed
    }
  }
  return null
}

export function readCameraMatrixFromUrl(): number[] | null {
  const related = getRelatedWindows()
  for (const target of related) {
    const parsed = parseCameraMatrixElements(
      (() => {
        try {
          const url = new URL(target.location.href)
          return url.searchParams.get(CAMERA_MATRIX_QUERY_KEY)
        } catch {
          return null
        }
      })()
    )
    if (parsed != null) {
      return parsed
    }
  }
  return null
}

export function writeCameraToUrl(components: readonly number[]): void {
  const related = getRelatedWindows()
  const serialized = serializeCameraComponents(components)
  for (const target of related) {
    try {
      const url = new URL(target.location.href)
      if (url.searchParams.get(CAMERA_QUERY_KEY) === serialized) {
        continue
      }
      url.searchParams.set(CAMERA_QUERY_KEY, serialized)
      target.history.replaceState(target.history.state, '', url.toString())
    } catch {
      // Ignore windows that cannot be updated.
    }
  }
}

export function writeCameraMatrixToUrl(elements: readonly number[]): void {
  const related = getRelatedWindows()
  const serialized = serializeCameraMatrixElements(elements)
  for (const target of related) {
    try {
      const url = new URL(target.location.href)
      if (url.searchParams.get(CAMERA_MATRIX_QUERY_KEY) === serialized) {
        continue
      }
      url.searchParams.set(CAMERA_MATRIX_QUERY_KEY, serialized)
      target.history.replaceState(target.history.state, '', url.toString())
    } catch {
      // Ignore windows that cannot be updated.
    }
  }
}

export function applyCameraMatrixToCamera(
  camera: Camera,
  elements: readonly number[]
): void {
  const matrix = new Matrix4().fromArray(Array.from(elements).slice(0, 16))
  const scale = new Vector3()
  matrix.decompose(camera.position, camera.quaternion, scale)
  camera.up.copy(WORLD_UP).applyQuaternion(camera.quaternion).normalize()
  camera.updateMatrixWorld(true)
}

export function cameraMatricesApproximatelyEqual(
  a: readonly number[] | null | undefined,
  b: readonly number[] | null | undefined,
  epsilon = 1e-6
): boolean {
  if (a == null || b == null || a.length !== 16 || b.length !== 16) {
    return false
  }
  for (let i = 0; i < 16; ++i) {
    if (Math.abs(a[i] - b[i]) > epsilon) {
      return false
    }
  }
  return true
}
