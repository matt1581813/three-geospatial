import { Matrix4, Ray, Vector3, type Camera } from 'three'

import { Ellipsoid, Geodetic, PointOfView, radians } from '@takram/three-geospatial'

export type CameraPointOfView = readonly [
  longitude: number,
  latitude: number,
  height: number,
  heading: number,
  pitch: number,
  distance: number
]

const RAD_TO_DEG = 180 / Math.PI
const DEG_TO_RAD = Math.PI / 180

const identityWorldToECEF = new Matrix4()
const rayScratch = new Ray()
const eyeWorldScratch = new Vector3()
const forwardWorldScratch = new Vector3()
const eyeECEFScratch = new Vector3()
const forwardECEFScratch = new Vector3()
const targetECEFScratch = new Vector3()
const eastScratch = new Vector3()
const northScratch = new Vector3()
const upScratch = new Vector3()
const directionECEFScratch = new Vector3()
const geodeticScratch = new Geodetic()
const targetFromTupleScratch = new Vector3()
const eyeFromTupleScratch = new Vector3()
const upFromTupleScratch = new Vector3()
const ecefToWorldScratch = new Matrix4()
const targetWorldScratch = new Vector3()
const eyeWorldFromTupleScratch = new Vector3()
const upWorldScratch = new Vector3()
const pointOfViewScratch = new PointOfView()

function clampUnit(value: number): number {
  return Math.min(1, Math.max(-1, value))
}

function getWorldToECEF(
  matrixWorldToECEF?: Matrix4 | null
): Matrix4 {
  return matrixWorldToECEF ?? identityWorldToECEF
}

export function readPointOfViewFromCamera(
  camera: Camera,
  matrixWorldToECEF?: Matrix4 | null,
  ellipsoid = Ellipsoid.WGS84
): CameraPointOfView | null {
  const worldToECEF = getWorldToECEF(matrixWorldToECEF)

  eyeWorldScratch.setFromMatrixPosition(camera.matrixWorld)
  forwardWorldScratch
    .set(0, 0, -1)
    .applyQuaternion(camera.quaternion)
    .normalize()

  eyeECEFScratch.copy(eyeWorldScratch).applyMatrix4(worldToECEF)
  forwardECEFScratch
    .copy(forwardWorldScratch)
    .transformDirection(worldToECEF)
    .normalize()

  const target = ellipsoid.getIntersection(
    rayScratch.set(eyeECEFScratch, forwardECEFScratch),
    targetECEFScratch
  )
  if (target == null) {
    return null
  }

  ellipsoid.getEastNorthUpVectors(
    target,
    eastScratch,
    northScratch,
    upScratch
  )

  directionECEFScratch.copy(target).sub(eyeECEFScratch).normalize()
  const heading =
    Math.atan2(
      northScratch.dot(directionECEFScratch),
      eastScratch.dot(directionECEFScratch)
    ) * RAD_TO_DEG
  const pitch =
    Math.asin(clampUnit(upScratch.dot(directionECEFScratch))) * RAD_TO_DEG
  const distance = eyeECEFScratch.distanceTo(target)

  const geodetic = geodeticScratch.setFromECEF(target)
  return [
    geodetic.longitude * RAD_TO_DEG,
    geodetic.latitude * RAD_TO_DEG,
    geodetic.height,
    heading,
    pitch,
    distance
  ]
}

export function applyPointOfViewToCamera(
  camera: Camera,
  pointOfView: readonly number[],
  matrixWorldToECEF?: Matrix4 | null,
  ellipsoid = Ellipsoid.WGS84
): boolean {
  if (pointOfView.length !== 6) {
    return false
  }
  const [longitude, latitude, height, heading, pitch, distance] = pointOfView
  if (
    ![
      longitude,
      latitude,
      height,
      heading,
      pitch,
      distance
    ].every(Number.isFinite)
  ) {
    return false
  }

  const worldToECEF = getWorldToECEF(matrixWorldToECEF)
  const ecefToWorld = ecefToWorldScratch.copy(worldToECEF).invert()

  new Geodetic(
    longitude * DEG_TO_RAD,
    latitude * DEG_TO_RAD,
    height
  ).toECEF(targetFromTupleScratch)
  pointOfViewScratch
    .set(distance, radians(heading), radians(pitch), 0)
    .decompose(
      targetFromTupleScratch,
      eyeFromTupleScratch,
      camera.quaternion,
      upFromTupleScratch,
      ellipsoid
    )

  targetWorldScratch.copy(targetFromTupleScratch).applyMatrix4(ecefToWorld)
  eyeWorldFromTupleScratch.copy(eyeFromTupleScratch).applyMatrix4(ecefToWorld)
  upWorldScratch
    .copy(upFromTupleScratch)
    .transformDirection(ecefToWorld)
    .normalize()

  camera.position.copy(eyeWorldFromTupleScratch)
  camera.up.copy(upWorldScratch)
  camera.lookAt(targetWorldScratch)
  camera.updateMatrixWorld(true)
  return true
}

export function pointOfViewsApproximatelyEqual(
  a: readonly number[] | null | undefined,
  b: readonly number[] | null | undefined,
  epsilon = 1e-5
): boolean {
  if (a == null || b == null || a.length !== 6 || b.length !== 6) {
    return false
  }
  for (let i = 0; i < 6; ++i) {
    if (Math.abs(a[i] - b[i]) > epsilon) {
      return false
    }
  }
  return true
}
